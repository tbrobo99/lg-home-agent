"""Tests for tier 3: the bounded tool loop.

Everything here runs against `tests/fake_llm.py`, so the suite is deterministic
and needs no API key. The interesting cases are the model being wrong — a
hallucinated device id, `"power": "false"` as a string, an unreadable time — not
the happy path.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import agent_loop, config, db, nlu  # noqa: E402
from app import agent as agent_mod  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.devices import MockAdapter  # noqa: E402
from app.devices import mock as mock_mod  # noqa: E402
from tests.fake_llm import FakeLLM, prose, tool_call  # noqa: E402


class LoopTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

        self._orig_data_dir = db.DATA_DIR
        self._orig_db_path = db.DB_PATH
        self._orig_state_path = mock_mod.STATE_PATH
        db.DATA_DIR = self.tmp
        db.DB_PATH = self.tmp / "test.db"
        mock_mod.STATE_PATH = self.tmp / "mock_state.json"
        db.init()

        self.fake = FakeLLM().__enter__()
        self.adapter = MockAdapter()
        self.cfg = dict(config.DEFAULTS)
        self.cfg["agent"] = {"default_room": "Living Room"}
        self.cfg["llm"] = {
            "enabled": True,
            "base_url": self.fake.base_url,
            "api_key": "test-key",
            "model": "fake",
            "timeout": 5,
        }
        self.now = datetime(2026, 10, 17, 21, 0)

    def tearDown(self) -> None:
        self.fake.__exit__(None, None, None)
        db.DATA_DIR = self._orig_data_dir
        db.DB_PATH = self._orig_db_path
        mock_mod.STATE_PATH = self._orig_state_path
        self._tmp.cleanup()

    def run_loop(self, text: str = "make it nicer in there") -> dict:
        return agent_loop.run(self.cfg, text, self.adapter, self.now, "Living Room")

    def state(self, device_id: str) -> dict:
        return self.adapter.get_state(device_id)


class ControlTests(LoopTestBase):
    def test_two_step_control_then_finish(self):
        self.fake.script = [
            tool_call("get_state", device_id="ac_bedroom"),
            tool_call("control", device_id="ac_bedroom",
                      actions=[{"action": "temperature", "value": 21}]),
            tool_call("finish", reply="Bedroom is set to 21."),
        ]
        result = self.run_loop()

        self.assertTrue(result["handled"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["reply"], "Bedroom is set to 21.")
        self.assertEqual(self.state("ac_bedroom")["target_temp"], 21)

    def test_string_power_is_coerced_not_rejected(self):
        """`glm-4-flash` really does send "false" instead of false."""
        self.fake.script = [
            tool_call("control", device_id="ac_living",
                      actions=[{"action": "power", "value": "false"}]),
            tool_call("finish", reply="Off."),
        ]
        result = self.run_loop()

        self.assertTrue(result["ok"], result)
        self.assertFalse(self.state("ac_living")["power"])
        self.assertEqual(result["steps"], 2)

    def test_hallucinated_device_id_is_recoverable(self):
        """The real failure: asked for the bedroom AC, the model said bedroom_ac."""
        self.fake.script = [
            tool_call("control", device_id="bedroom_ac",
                      actions=[{"action": "power", "value": True}]),
            tool_call("control", device_id="ac_bedroom",
                      actions=[{"action": "power", "value": True}]),
            tool_call("finish", reply="Bedroom AC on."),
        ]
        result = self.run_loop()

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["steps"], 3)
        self.assertTrue(self.state("ac_bedroom")["power"])

        # the error observation must have reached the model
        second = self.fake.requests[1]["messages"]
        self.assertTrue(any("no device with id" in str(m.get("content"))
                            for m in second if m.get("role") == "tool"))

    def test_offline_device_is_refused(self):
        devices = self.adapter.list_devices()
        target = next(d for d in devices if d["id"] == "ac_bedroom")
        original = self.adapter.list_devices

        def offline():
            out = original()
            for d in out:
                if d["id"] == target["id"]:
                    d["online"] = False
            return out

        self.adapter.list_devices = offline
        self.fake.script = [
            tool_call("control", device_id="ac_bedroom",
                      actions=[{"action": "power", "value": True}]),
            tool_call("finish", reply="It is offline."),
        ]
        result = self.run_loop()

        self.assertFalse(result["ok"])
        self.assertFalse(self.state("ac_bedroom")["power"])
        self.adapter.list_devices = original


class ScheduleTests(LoopTestBase):
    def test_readable_time_creates_a_job(self):
        self.fake.script = [
            tool_call("schedule", device_id="ac_bedroom",
                      actions=[{"action": "power", "value": False}],
                      when_text="every night at 11 pm"),
            tool_call("finish", reply="Scheduled."),
        ]
        result = self.run_loop()

        self.assertTrue(result["ok"], result)
        jobs = db.list_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["device_id"], "ac_bedroom")
        self.assertEqual(jobs[0]["rule"]["type"], "daily")
        self.assertEqual(jobs[0]["rule"]["at"], "23:00")

    def test_unreadable_time_is_refused_and_asked_about(self):
        self.fake.script = [
            tool_call("schedule", device_id="ac_bedroom",
                      actions=[{"action": "power", "value": False}],
                      when_text="whenever i feel like it"),
            tool_call("ask_user", question="When should it switch off?"),
        ]
        result = self.run_loop()

        self.assertEqual(db.list_jobs(), [])
        self.assertTrue(result["awaiting"])
        self.assertEqual(result["reply"], "When should it switch off?")
        # the refusal must be visible to the model, not swallowed
        second = self.fake.requests[1]["messages"]
        self.assertTrue(any("could not read a time" in str(m.get("content"))
                            for m in second if m.get("role") == "tool"))

    def test_model_cannot_smuggle_a_timestamp(self):
        """The schedule tool has no timestamp field, so a model that tries to
        supply one gets an error instead of a job."""
        self.fake.script = [
            tool_call("schedule", device_id="ac_bedroom",
                      actions=[{"action": "power", "value": False}],
                      when_text="11 pm", run_at="2026-10-17T23:30:00"),
            tool_call("ask_user", question="When exactly?"),
        ]
        self.run_loop()

        self.assertEqual(db.list_jobs(), [])
        second = self.fake.requests[1]["messages"]
        self.assertTrue(any("bad arguments" in str(m.get("content"))
                            for m in second if m.get("role") == "tool"))


class GuardrailTests(LoopTestBase):
    def test_step_budget_stops_a_loop_that_never_finishes(self):
        self.fake.script = [
            tool_call("get_state", device_id="ac_living") for _ in range(10)
        ]
        result = self.run_loop()

        self.assertEqual(result["steps"], agent_loop.MAX_STEPS)
        self.assertFalse(result["handled"])

    def test_unknown_tool_is_reported_back(self):
        self.fake.script = [
            tool_call("teleport", device_id="ac_living"),
            tool_call("finish", reply="Cannot do that."),
        ]
        result = self.run_loop()

        self.assertTrue(result["handled"])
        second = self.fake.requests[1]["messages"]
        self.assertTrue(any("no such tool" in str(m.get("content"))
                            for m in second if m.get("role") == "tool"))

    def test_out_of_range_temperature_is_refused(self):
        self.fake.script = [
            tool_call("control", device_id="ac_living",
                      actions=[{"action": "temperature", "value": 99}]),
            tool_call("finish", reply="Too hot."),
        ]
        result = self.run_loop()

        self.assertFalse(result["ok"])
        self.assertNotEqual(self.state("ac_living")["target_temp"], 99)

    def test_no_api_key_never_calls_out(self):
        self.cfg["llm"]["api_key"] = ""
        result = self.run_loop()
        self.assertFalse(result["handled"])
        self.assertEqual(self.fake.requests, [])


class CascadeTests(LoopTestBase):
    def test_agent_reaches_tier_three_when_the_parser_fails(self):
        self.fake.script = [
            tool_call("list_devices"),
            tool_call("control", device_id="purifier",
                      actions=[{"action": "power", "value": True}]),
            tool_call("finish", reply="Purifier is running."),
        ]
        agent = Agent(self.adapter, self.cfg)
        result = agent.handle("something is making the air awful in here")

        self.assertEqual(result["intent_name"], "agent_loop")
        self.assertTrue(result["ok"])
        self.assertTrue(self.state("purifier")["power"])

    def test_ask_then_answer_resumes_the_same_thread(self):
        self.fake.script = [
            tool_call("ask_user", question="Which room?"),
            tool_call("control", device_id="ac_bedroom",
                      actions=[{"action": "power", "value": True}]),
            tool_call("finish", reply="Bedroom AC on."),
        ]
        agent = Agent(self.adapter, self.cfg)

        first = agent.handle("make it nicer in there")
        self.assertTrue(first["reply"].endswith("Which room?"))

        second = agent.handle("the bedroom one")
        self.assertEqual(second["intent_name"], "agent_loop")
        self.assertTrue(self.state("ac_bedroom")["power"])

        # the resumed call must carry the question it is answering
        resumed = self.fake.requests[-1]["messages"]
        self.assertTrue(any("Which room?" in str(m.get("content")) for m in resumed))

    def test_a_new_command_does_not_resume_an_old_thread(self):
        self.fake.script = [tool_call("ask_user", question="Which room?")]
        agent = Agent(self.adapter, self.cfg)
        agent.handle("make it nicer in there")

        # a complete command must drop the thread, not answer it
        result = agent.handle("turn on the balcony washer")
        self.assertEqual(result["intent_name"], "control")
        self.assertTrue(self.state("washer")["power"])


class ProseBeforeActingTests(LoopTestBase):
    """A model that talks instead of calling a tool has not done anything.

    This is the failure that mattered most in practice: `glm-4-flash` answered
    "the air in here is disgusting" with "would you like me to turn on the
    purifier?" and "get the flat ready for bed" with "the request is not clear",
    which then became the user-facing reply. The loop nudges once.
    """

    def test_prose_before_any_tool_gets_one_nudge(self):
        self.fake.script = [
            prose("Would you like me to turn on the purifier?"),
            tool_call("control", device_id="purifier",
                      actions=[{"action": "power", "value": True}]),
            tool_call("finish", reply="Purifier is on."),
        ]
        result = self.run_loop("the air in here is disgusting")

        self.assertTrue(result["ok"], result)
        self.assertTrue(self.state("purifier")["power"])
        self.assertEqual(result["reply"], "Purifier is on.")

        # the nudge must reach the model, and only once
        nudged = self.fake.requests[1]["messages"]
        self.assertTrue(any("did not call a tool" in str(m.get("content"))
                            for m in nudged))

    def test_prose_twice_is_accepted_as_the_answer(self):
        """Some questions need no tool ("what can you do?"), so the nudge is a
        single retry, not a loop that never lets the model speak."""
        self.fake.script = [
            prose("I can control the AC, purifier, washer and fridge."),
            prose("I can control the AC, purifier, washer and fridge."),
        ]
        result = self.run_loop("what sort of things can you do around here")

        self.assertTrue(result["handled"])
        self.assertIn("I can control", result["reply"])
        self.assertEqual(result["steps"], 2)

    def test_prose_after_a_tool_is_accepted_immediately(self):
        """Once something has actually happened, a closing sentence is a reply."""
        self.fake.script = [
            tool_call("control", device_id="purifier",
                      actions=[{"action": "power", "value": True}]),
            prose("The purifier is running now."),
        ]
        result = self.run_loop("the air in here is disgusting")

        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["reply"], "The purifier is running now.")
        self.assertTrue(self.state("purifier")["power"])

    def test_a_tool_name_pasted_into_prose_is_stripped(self):
        """Seen end-to-end: the model wrote "finish\\nThe bedroom AC has been…"
        as prose instead of calling the tool, and the word reached the user."""
        self.fake.script = [
            tool_call("control", device_id="purifier",
                      actions=[{"action": "power", "value": True}]),
            prose("finish\nThe purifier is running now."),
        ]
        result = self.run_loop("the air in here is disgusting")

        self.assertEqual(result["reply"], "The purifier is running now.")


class SummaryTests(LoopTestBase):
    """The loop's own words, when the model stops without calling `finish`.

    These are what a user reads, so the shapes are pinned: an earlier version
    flattened a whole-home request into "turn off, turn off, turn off".
    """

    def test_whole_home_summary_names_every_device(self):
        self.fake.script = [
            tool_call("control", device_id="ac_living",
                      actions=[{"action": "power", "value": False}]),
            tool_call("control", device_id="ac_bedroom",
                      actions=[{"action": "power", "value": False}]),
            tool_call("control", device_id="purifier",
                      actions=[{"action": "power", "value": False}]),
        ]
        result = self.run_loop()

        self.assertTrue(result["handled"], result)
        self.assertEqual(
            result["reply"],
            "Done \u2014 Living Room AC, Bedroom AC and Living Room Purifier: off.")

    def test_different_actions_get_a_clause_each(self):
        self.fake.script = [
            tool_call("control", device_id="ac_living",
                      actions=[{"action": "power", "value": False}]),
            tool_call("control", device_id="purifier",
                      actions=[{"action": "power", "value": True}]),
        ]
        result = self.run_loop()

        self.assertEqual(
            result["reply"],
            "Done \u2014 Living Room AC off; Living Room Purifier on.")

    def test_each_action_says_which_device_it_belongs_to(self):
        """Without attribution, four identical `power:false` entries in the
        result are indistinguishable from one repeated four times."""
        self.fake.script = [
            tool_call("control", device_id="ac_living",
                      actions=[{"action": "power", "value": False}]),
            tool_call("control", device_id="ac_bedroom",
                      actions=[{"action": "power", "value": False}]),
        ]
        result = self.run_loop()

        self.assertEqual([a["device_id"] for a in result["actions"]],
                         ["ac_living", "ac_bedroom"])


class RoutingTests(LoopTestBase):
    """Tier 2 is blind: one message, a device list, one device, no state.

    So the router must not hand it anything whose answer depends on state it
    cannot see. These tests pin that decision, because a router that quietly
    lets tier 2 answer everything makes tier 3 unreachable — and tier 3 is the
    reason the cascade has three tiers at all.

    "Reaches the loop" is asserted on the request body: a tier 2 call carries
    `response_format`, a tier 3 call carries `tools`.
    """

    def test_a_symptom_skips_the_blind_pass(self):
        self.fake.script = [tool_call("finish", reply="Done.")]
        result = Agent(self.adapter, self.cfg).handle(
            "the bedroom is a sauna, sort it out")

        self.assertEqual(result["tier"], "loop")
        first = self.fake.requests[0]
        self.assertIn("tools", first)
        self.assertNotIn("response_format", first)

    def test_a_whole_home_request_skips_the_blind_pass(self):
        """One device is all tier 2 can express, so a whole-home request must
        not be answered by it — it would do one room and report success."""
        self.fake.script = [tool_call("finish", reply="Done.")]
        Agent(self.adapter, self.cfg).handle("the whole house is too warm")
        self.assertIn("tools", self.fake.requests[0])

    def test_the_blind_pass_is_never_asked_about_the_whole_home(self):
        """Same decision, isolated: the text names a value ("22") so the setting
        gate is satisfied, which leaves scope as the only reason to refuse."""
        unknown = {"intent": "unknown", "device_id": None, "actions": [],
                   "rule": None, "clarify": None}
        agent = Agent(self.adapter, self.cfg)

        with mock.patch.object(agent_mod.llm, "plan") as plan:
            plan.return_value = None
            agent._escalate("turn everything to 22", unknown,
                            self.adapter.list_devices(), "Living Room", self.now)
            plan.assert_not_called()

    def test_a_half_parsed_request_reaches_the_loop(self):
        """"put the living room to sleep in 20 minutes" is `need_action`: the
        rules found a time but no verb. Asking the user is the last resort."""
        self.fake.script = [tool_call("finish", reply="Switching it off at 21:20.")]
        result = Agent(self.adapter, self.cfg).handle(
            "put the living room to sleep in 20 minutes")

        self.assertEqual(result["intent_name"], "agent_loop")
        self.assertIn("tools", self.fake.requests[0])

    def test_a_half_parsed_request_still_asks_when_the_loop_gives_up(self):
        """With no model help available the canned question is the honest floor."""
        self.fake.script = []
        result = Agent(self.adapter, self.cfg).handle("it's stuffy in here")
        self.assertEqual(result["intent_name"], "need_target")

    def test_a_complete_blind_plan_is_accepted(self):
        """The one shape tier 2 is good for: the user named the value, so the
        only open question is which device — and a list of devices answers it."""
        plan = {"intent": "control", "device_id": "ac_bedroom",
                "device_name": "Bedroom AC",
                "actions": [{"action": "temperature", "value": 24}],
                "rule": None, "repeat_label": "", "confidence": 0.8,
                "clarify": None, "raw_text": "", "source": "llm"}
        unknown = {"intent": "unknown", "device_id": None, "actions": [],
                   "rule": None, "clarify": None}
        agent = Agent(self.adapter, self.cfg)

        with mock.patch.object(agent_mod.llm, "plan", return_value=plan):
            got = agent._escalate("set the bedroom to 24", unknown,
                                  self.adapter.list_devices(), "Living Room",
                                  self.now)
        self.assertIs(got, plan)

    def test_an_incomplete_blind_plan_is_rejected(self):
        """A plan that names a device but no action is a cheap answer plus a
        guess — and the guess is the part the loop does better."""
        plan = {"intent": "control", "device_id": "ac_bedroom",
                "device_name": "Bedroom AC", "actions": [], "rule": None,
                "repeat_label": "", "confidence": 0.8, "clarify": None,
                "raw_text": "", "source": "llm"}
        unknown = {"intent": "unknown", "device_id": None, "actions": [],
                   "rule": None, "clarify": None}
        agent = Agent(self.adapter, self.cfg)

        with mock.patch.object(agent_mod.llm, "plan", return_value=plan):
            got = agent._escalate("set the bedroom to 24", unknown,
                                  self.adapter.list_devices(), "Living Room",
                                  self.now)
        self.assertIsNone(got)


class RuleTierIsUntouchedTests(LoopTestBase):
    def test_rules_still_answer_without_any_http_call(self):
        agent = Agent(self.adapter, self.cfg)
        result = agent.handle("turn off the living room ac")
        self.assertEqual(result["intent_name"], "control")
        self.assertEqual(result["tier"], "rules")
        self.assertEqual(self.fake.requests, [])
        self.assertFalse(self.state("ac_living")["power"])


if __name__ == "__main__":
    unittest.main()
