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


if __name__ == "__main__":
    unittest.main()
