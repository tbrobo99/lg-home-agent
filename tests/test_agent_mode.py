"""Agent mode: the rules offer, the model decides.

The cascade these tests sit next to is a workflow — `_escalate` reads the shape
of the sentence in code and picks the tier, so the model never gets to say the
rules were wrong. Agent mode keeps every guardrail and inverts that one
relationship: the parse is handed to the loop as an offer, and `adopt_parse` is
how the model takes it.

So the four things worth asserting here are all about *choice*:

- the offer is in the prompt, in the model's terms;
- accepting it behaves exactly as the rules would have;
- rejecting it and planning something else actually plans something else;
- turning the mode off leaves the deterministic path untouched.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import agent_loop, config, db  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.devices import MockAdapter  # noqa: E402
from app.devices import mock as mock_mod  # noqa: E402
from tests.fake_llm import FakeLLM, tool_call  # noqa: E402


class AgentModeTestBase(unittest.TestCase):
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
        self.cfg["agent"] = {"default_room": "Living Room", "mode": "agent"}
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
        self.adapter.stop()
        db.DATA_DIR = self._orig_data_dir
        db.DB_PATH = self._orig_db_path
        mock_mod.STATE_PATH = self._orig_state_path
        self._tmp.cleanup()

    # ------------------------------------------------ helpers

    def ask(self, text: str) -> dict:
        return Agent(self.adapter, self.cfg).handle(text)

    def power(self, device_id: str):
        return self.adapter.get_state(device_id).get("power")

    def turn_on(self, device_id: str) -> None:
        """The simulator starts with the living room AC off, and "still off"
        would make a test that asserts nothing happened pass by accident."""
        self.adapter.apply(device_id, [{"action": "power", "value": True}])

    def system_prompt(self) -> str:
        """The prompt the model actually saw, offer and all."""
        for req in self.fake.requests:
            for m in req.get("messages") or []:
                if m.get("role") == "system":
                    return str(m.get("content") or "")
        return ""


class OfferTests(AgentModeTestBase):
    def test_the_offer_states_what_the_rules_would_do(self):
        self.fake.script = [tool_call("adopt_parse")]
        self.ask("turn off the living room AC")

        prompt = self.system_prompt()
        self.assertIn("RULE ENGINE'S READING", prompt)
        self.assertIn("intent: control", prompt)
        self.assertIn("Living Room AC", prompt)
        self.assertIn("turn off", prompt)

    def test_the_offer_is_worded_as_a_suggestion(self):
        self.fake.script = [tool_call("adopt_parse")]
        self.ask("turn off the living room AC")

        prompt = self.system_prompt()
        self.assertIn("not an instruction", prompt)
        self.assertIn("not obliged to agree", prompt)

    def test_a_question_outside_the_house_is_not_offered(self):
        """A refusal about what this house can see is not a judgement to revise.

        Letting the model revise it is how a weather forecast gets invented, so
        agent mode hands it back to the rules untouched.
        """
        self.ask("what is the weather like tomorrow")
        self.assertEqual(self.fake.requests, [])


class AdoptTests(AgentModeTestBase):
    def test_accepting_the_parse_does_what_the_rules_would_have(self):
        self.turn_on("ac_living")
        self.fake.script = [tool_call("adopt_parse")]
        result = self.ask("turn off the living room AC")

        self.assertTrue(result["adopted"])
        self.assertEqual(result["tier"], "agent")
        self.assertIs(self.power("ac_living"), False)
        self.assertIn("Living Room AC", result["reply"])

    def test_accepting_leaves_one_step_in_the_trace(self):
        self.fake.script = [tool_call("adopt_parse")]
        result = self.ask("turn off the living room AC")

        self.assertEqual(len(result["trace"]), 1)
        self.assertTrue(result["trace"][0]["result"])

    def test_a_lookup_is_not_put_in_front_of_the_model(self):
        """A storage answer has no tool and no decision in it.

        Measured against a real model: it ran a full loop over a question it
        had no data for, gave up, and the rules answered anyway — eight seconds
        added to a one-millisecond answer. Agent mode is for decisions, and
        "how long does chicken keep" is not one.
        """
        self.fake.script = [tool_call("adopt_parse")]
        result = self.ask("how long can I keep the chicken")

        self.assertEqual(self.fake.requests, [])
        self.assertIn("chicken", result["reply"].lower())

    def test_the_task_list_is_not_put_in_front_of_the_model(self):
        self.fake.script = [tool_call("adopt_parse")]
        self.ask("what scheduled tasks do I have")

        self.assertEqual(self.fake.requests, [])


class RejectTests(AgentModeTestBase):
    def test_the_model_may_plan_something_else(self):
        """The offer said the living room; the model sends the bedroom."""
        self.turn_on("ac_living")
        self.fake.script = [
            tool_call("control", device_id="ac_bedroom",
                      actions=[{"action": "power", "value": False}]),
            tool_call("finish", reply="Done — I switched the Bedroom AC off."),
        ]
        result = self.ask("turn off the living room AC")

        self.assertFalse(result["adopted"])
        self.assertIs(self.power("ac_bedroom"), False)
        self.assertIs(self.power("ac_living"), True)
        self.assertEqual(len(result["trace"]), 2)

    def test_the_trace_names_the_rejected_offer_first(self):
        self.fake.script = [
            tool_call("control", device_id="ac_bedroom",
                      actions=[{"action": "power", "value": False}]),
            tool_call("finish", reply="Done."),
        ]
        result = self.ask("turn off the living room AC")

        self.assertEqual([s["tool"] for s in result["trace"]],
                         ["control", "finish"])


class FallbackTests(AgentModeTestBase):
    def test_a_model_that_does_nothing_leaves_the_rules_in_charge(self):
        """Agent mode may not make the system less reliable than the cascade."""
        self.turn_on("ac_living")
        self.fake.script = []
        result = self.ask("turn off the living room AC")

        self.assertIs(self.power("ac_living"), False)
        self.assertEqual(result["tier"], "rules")

    def test_cascade_mode_never_calls_a_model(self):
        self.turn_on("ac_living")
        self.cfg["agent"]["mode"] = "cascade"
        self.fake.script = [tool_call("adopt_parse")]
        result = self.ask("turn off the living room AC")

        self.assertEqual(self.fake.requests, [])
        self.assertIs(self.power("ac_living"), False)
        self.assertNotEqual(result["tier"], "agent")


class TraceTests(AgentModeTestBase):
    def test_a_failed_step_stays_in_the_trace(self):
        """Recovering from itself is the loop's whole advantage; show it."""
        self.fake.script = [
            tool_call("control", device_id="ac_kitchen",
                      actions=[{"action": "power", "value": False}]),
            tool_call("adopt_parse"),
        ]
        result = self.ask("turn off the living room AC")

        self.assertEqual(len(result["trace"]), 2)
        self.assertIn("ac_kitchen", result["trace"][0]["result"])
        self.assertTrue(result["adopted"])

    def test_the_offer_survives_a_schedule(self):
        self.fake.script = [tool_call("adopt_parse")]
        self.ask("turn off the living room AC every night at 11 pm")

        prompt = self.system_prompt()
        self.assertIn("intent: schedule", prompt)
        self.assertIn("THIS REQUEST HAS A TIME", prompt)


if __name__ == "__main__":
    unittest.main()
