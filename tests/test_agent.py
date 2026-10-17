"""End-to-end tests for the agent, the simulator, and the scheduler.

Uses a temporary database and a temporary simulator state file, so the real
data/ directory is never touched.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, db  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.devices import MockAdapter  # noqa: E402
from app.devices import mock as mock_mod  # noqa: E402
from app.scheduler import Scheduler  # noqa: E402


class AgentTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

        # isolate persistence
        self._orig_data_dir = db.DATA_DIR
        self._orig_db_path = db.DB_PATH
        self._orig_state_path = mock_mod.STATE_PATH
        db.DATA_DIR = self.tmp
        db.DB_PATH = self.tmp / "test.db"
        mock_mod.STATE_PATH = self.tmp / "mock_state.json"

        db.init()
        self.adapter = MockAdapter()
        cfg = dict(config.DEFAULTS)
        cfg["agent"] = {"default_room": "Living Room"}
        self.agent = Agent(self.adapter, cfg)

    def tearDown(self):
        self.adapter.stop()
        db.DATA_DIR = self._orig_data_dir
        db.DB_PATH = self._orig_db_path
        mock_mod.STATE_PATH = self._orig_state_path
        self._tmp.cleanup()


class ControlTest(AgentTestBase):
    def test_turn_on_reports_state(self):
        r = self.agent.handle("turn on the living room AC")
        self.assertTrue(r["ok"])
        self.assertIn("Living Room AC is on", r["reply"])
        self.assertTrue(self.adapter.get_state("ac_living")["power"])

    def test_reference_resolves_to_last_device(self):
        self.agent.handle("turn on the bedroom AC")
        r = self.agent.handle("set it to 22 degrees")
        self.assertTrue(r["ok"])
        self.assertEqual(self.adapter.get_state("ac_bedroom")["target_temp"], 22)

    def test_unsupported_capability_is_rejected(self):
        r = self.agent.handle("set the purifier to 24 degrees")
        self.assertFalse(r["ok"])
        self.assertIn("does not support", r["reply"])

    def test_temperature_is_clamped_by_device(self):
        self.agent.handle("turn on the living room AC")
        self.agent.handle("set it to 30 degrees")
        self.assertEqual(self.adapter.get_state("ac_living")["target_temp"], 30)


class QueryTest(AgentTestBase):
    def test_query_off_device(self):
        r = self.agent.handle("what's the temperature in the bedroom?")
        self.assertTrue(r["ok"])
        self.assertIn("Bedroom AC", r["reply"])

    def test_query_after_turning_on(self):
        self.agent.handle("turn on the living room AC")
        r = self.agent.handle("what's the temperature?")
        self.assertIn("running", r["reply"])


class ScheduleTest(AgentTestBase):
    def test_daily_job_is_persisted(self):
        r = self.agent.handle("turn off the AC every night at 11 pm")
        self.assertTrue(r["ok"])
        jobs = db.list_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["rule"], {"type": "daily", "at": "23:00"})
        self.assertEqual(jobs[0]["device_id"], "ac_living")

    def test_list_and_cancel(self):
        self.agent.handle("turn off the AC every night at 11 pm")
        r = self.agent.handle("what scheduled tasks do I have?")
        self.assertIn("scheduled task", r["reply"])
        r = self.agent.handle("cancel all scheduled tasks")
        self.assertIn("Cleared", r["reply"])
        self.assertEqual(db.list_jobs(), [])

    def test_cancel_specific_device(self):
        self.agent.handle("turn off the AC every night at 11 pm")
        self.agent.handle("turn off the washer at 11 pm")
        r = self.agent.handle("cancel the washer schedule")
        self.assertTrue(r["ok"])
        self.assertEqual(len(db.list_jobs()), 1)


class SchedulerTest(AgentTestBase):
    def _make_job(self, rule: dict, actions=None, seconds_ahead=0):
        run_at = (datetime.now() + timedelta(seconds=seconds_ahead)).isoformat(timespec="seconds")
        rule = dict(rule)
        if rule["type"] == "once":
            rule["run_at"] = run_at
        return db.add_job("washer", "Balcony Washer",
                          actions or [{"action": "power", "value": True}],
                          rule, run_at, "test", "test")

    def test_one_time_job_runs_and_disables(self):
        job_id = self._make_job({"type": "once"})
        sched = Scheduler(self.adapter)
        sched._run_job(db.get_job(job_id), datetime.now())

        job = db.get_job(job_id)
        self.assertEqual(job["enabled"], 0)
        self.assertIn("Success", job["last_result"])
        self.assertTrue(self.adapter.get_state("washer")["power"])

    def test_recurring_job_stays_enabled_and_advances(self):
        job_id = self._make_job({"type": "daily", "at": "23:00"})
        before = db.get_job(job_id)["next_run_at"]
        sched = Scheduler(self.adapter)
        sched._run_job(db.get_job(job_id), datetime.now())

        job = db.get_job(job_id)
        self.assertEqual(job["enabled"], 1)
        self.assertNotEqual(job["next_run_at"], before)

    def test_run_is_pushed_into_chat(self):
        job_id = self._make_job({"type": "once"})
        Scheduler(self.adapter)._run_job(db.get_job(job_id), datetime.now())
        messages = db.recent_messages(5)
        self.assertTrue(any(m["role"] == "system" for m in messages))

    def test_offline_device_is_recorded_as_failure(self):
        job_id = self._make_job({"type": "once"}, actions=[{"action": "power", "value": True}])
        self.adapter.list_devices = lambda: []  # pretend the device vanished
        Scheduler(self.adapter)._run_job(db.get_job(job_id), datetime.now())
        self.assertIn("Failed", db.get_job(job_id)["last_result"])


class GroupControlTest(AgentTestBase):
    def test_room_level_command_hits_every_device_in_it(self):
        self.agent.handle("turn on the living room AC")
        self.agent.handle("turn on the purifier")
        r = self.agent.handle("turn off everything in the living room")
        self.assertTrue(r["ok"])
        self.assertFalse(self.adapter.get_state("ac_living")["power"])
        self.assertFalse(self.adapter.get_state("purifier")["power"])
        self.assertIn("2 of 2", r["reply"])

    def test_partial_failure_is_reported_per_device(self):
        # the purifier has no temperature capability
        r = self.agent.handle("set everything in the living room to 24 degrees")
        self.assertIn("1 of 2", r["reply"])
        self.assertIn("skipped", r["reply"])
        self.assertEqual(self.adapter.get_state("ac_living")["target_temp"], 24)

    def test_skip_reason_does_not_repeat_the_device_name(self):
        r = self.agent.handle("set everything in the living room to 24 degrees")
        purifier_line = next(
            ln for ln in r["reply"].splitlines() if "Purifier" in ln)
        self.assertEqual(purifier_line.count("Purifier"), 1, purifier_line)
        self.assertIn("does not support", purifier_line)

    def test_group_schedule_creates_one_job_per_device(self):
        self.agent.handle("turn off everything in the living room at 11 pm")
        jobs = db.list_jobs()
        self.assertEqual(len(jobs), 2)
        self.assertEqual({j["device_id"] for j in jobs}, {"ac_living", "purifier"})


class DialogueStateTest(AgentTestBase):
    def test_comfort_without_device_asks_then_completes(self):
        r = self.agent.handle("it's stuffy in here")
        self.assertEqual(r["intent_name"], "need_target")
        self.assertIn("Which", r["reply"])

        r2 = self.agent.handle("bedroom")
        self.assertTrue(r2["ok"])
        state = self.adapter.get_state("ac_bedroom")
        self.assertTrue(state["power"])
        self.assertEqual(state["mode"], "cool")
        self.assertEqual(state["target_temp"], 24)

    def test_missing_action_is_filled_by_next_turn(self):
        r = self.agent.handle("set a timer for the washer at 11 pm")
        self.assertEqual(r["intent_name"], "need_action")

        r2 = self.agent.handle("turn it off")
        self.assertTrue(r2["ok"])
        jobs = db.list_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["device_id"], "washer")
        self.assertEqual(jobs[0]["actions"], [{"action": "power", "value": False}])

    def test_pending_request_can_be_cancelled(self):
        self.agent.handle("it's stuffy in here")
        r = self.agent.handle("never mind")
        self.assertTrue(r["ok"])
        self.assertIn("dropped", r["reply"].lower())

    def test_pending_state_is_not_leaked_into_unrelated_turns(self):
        self.agent.handle("it's stuffy in here")
        self.agent.handle("what's the temperature in the bedroom?")
        # the pending comfort request must not hijack an unrelated question
        self.assertFalse(self.adapter.get_state("ac_bedroom")["power"])


class CatchUpTest(AgentTestBase):
    def test_overdue_one_time_job_is_closed_not_fired(self):
        past = (datetime.now() - timedelta(hours=3)).isoformat(timespec="seconds")
        job_id = db.add_job("washer", "Balcony Washer",
                            [{"action": "power", "value": True}],
                            {"type": "once", "run_at": past}, past, "past", "test")

        Scheduler(self.adapter)._catch_up()

        job = db.get_job(job_id)
        self.assertEqual(job["enabled"], 0)
        self.assertIn("Missed", job["last_result"])
        self.assertFalse(self.adapter.get_state("washer")["power"])

    def test_overdue_recurring_job_is_rescheduled(self):
        past = (datetime.now() - timedelta(days=2)).isoformat(timespec="seconds")
        job_id = db.add_job("ac_living", "Living Room AC",
                            [{"action": "power", "value": False}],
                            {"type": "daily", "at": "07:00"}, past, "daily", "test")

        Scheduler(self.adapter)._catch_up()

        job = db.get_job(job_id)
        self.assertEqual(job["enabled"], 1)
        self.assertIn("Missed", job["last_result"])
        self.assertGreater(job["next_run_at"],
                           datetime.now().isoformat(timespec="seconds"))

    def test_future_jobs_are_untouched(self):
        future = (datetime.now() + timedelta(hours=2)).isoformat(timespec="seconds")
        job_id = db.add_job("ac_living", "Living Room AC",
                            [{"action": "power", "value": False}],
                            {"type": "once", "run_at": future}, future, "later", "test")

        Scheduler(self.adapter)._catch_up()

        job = db.get_job(job_id)
        self.assertEqual(job["enabled"], 1)
        self.assertEqual(job["next_run_at"], future)


if __name__ == "__main__":
    unittest.main()
