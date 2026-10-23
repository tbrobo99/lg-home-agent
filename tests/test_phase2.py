"""Phase 2 tests: sensors, risk indices, risk-graded policy, approvals, and the
proactive engine end to end.

Uses a temporary database (like test_agent.py) and a scripted FakeLLM, so the
real data/ directory is never touched and CI needs no network.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db, policy, risk  # noqa: E402
from app.approval import ApprovalQueue  # noqa: E402
from app.proactive import ProactiveEngine  # noqa: E402
from tests.fake_llm import FakeLLM  # noqa: E402


class FakeAdapter:
    """Records what the engine tried to do; executes nothing real."""

    backend_name = "fake"

    def __init__(self):
        self.applied: list[tuple[str, list]] = []
        self.devices = [
            {"id": "range_hood", "name": "Kitchen Range Hood", "type": "range_hood",
             "room": "Kitchen", "online": True, "capabilities": ["power", "fan"]},
            {"id": "washer", "name": "Balcony Washer", "type": "washer",
             "room": "Balcony", "online": True, "capabilities": ["power", "ventilate"]},
            {"id": "dryer", "name": "Balcony Dryer", "type": "dryer",
             "room": "Balcony", "online": True, "capabilities": ["power"]},
            {"id": "dehumidifier", "name": "Balcony Dehumidifier", "type": "dehumidifier",
             "room": "Balcony", "online": True, "capabilities": ["power", "fan"]},
            {"id": "oven", "name": "Kitchen Oven", "type": "oven",
             "room": "Kitchen", "online": True, "capabilities": ["power", "temperature"]},
        ]

    def list_devices(self):
        return list(self.devices)

    def get_device(self, device_id):
        return next((d for d in self.devices if d["id"] == device_id), None)

    def get_state(self, device_id):
        return {"power": False}

    def apply(self, device_id, actions):
        self.applied.append((device_id, [dict(a) for a in actions]))
        return {"ok": True, "applied": actions, "error": None}


def _cfg(llm=None, **proactive):
    cfg = {"llm": llm or {"enabled": False},
           "proactive": {"tick_seconds": 0.05, "cooldown_seconds": 0, **proactive}}
    return cfg


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = (db.DATA_DIR, db.DB_PATH)
        db.DATA_DIR = Path(self._tmp.name)
        db.DB_PATH = db.DATA_DIR / "test.db"
        db.init()
        self.adapter = FakeAdapter()

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH = self._orig
        self._tmp.cleanup()

    def _engine(self, cfg=None):
        e = ProactiveEngine(self.adapter, cfg or _cfg())
        e.sensors.write_files = False  # keep the repo's data/ clean
        return e


# ---------------------------------------------------------------- risk indices

class RiskTest(Base):
    def _snapshot(self, scenario, now=None):
        e = self._engine()
        e.sensors.apply_scenario(scenario, now=now)
        return e.sensors.snapshot(), e.sensors.history

    def test_mold_risk_fires_on_sealed_wet_drum(self):
        snap, hist = self._snapshot("mold")
        r = risk.mold_risk(snap, hist)
        self.assertIsNotNone(r)
        self.assertGreaterEqual(r.score, r.threshold)
        self.assertEqual(r.kind, "mold")
        # rising drum -> the 30-minute prediction is worse than now
        self.assertGreaterEqual(r.predicted, r.current)
        self.assertIn("sealed_hours", r.drivers)

    def test_mold_risk_silent_when_door_open(self):
        snap, hist = self._snapshot("mold")
        snap["appliances"]["washer"]["door_closed"] = False
        self.assertIsNone(risk.mold_risk(snap, hist))

    def test_mold_risk_silent_when_no_cycle(self):
        snap, hist = self._snapshot("mold")
        snap["appliances"]["washer"]["cycle"] = "standby"
        self.assertIsNone(risk.mold_risk(snap, hist))

    def test_thermal_risk_fires_with_dryer_and_scheduled_oven(self):
        now = datetime.now()
        snap, hist = self._snapshot("oven", now=now)
        r = risk.thermal_risk(snap, hist, now)
        self.assertIsNotNone(r)
        self.assertGreaterEqual(r.score, r.threshold)
        self.assertEqual(r.kind, "thermal")
        self.assertEqual(r.drivers["heat_load_w"], 1800)
        self.assertIsNotNone(r.drivers["oven_scheduled_in_min"])

    def test_thermal_risk_silent_when_no_heat(self):
        snap, hist = self._snapshot("idle")
        self.assertIsNone(risk.thermal_risk(snap, hist))

    def test_slope_of_rising_series_is_positive(self):
        t0 = datetime.now()
        series = deque((t0 + timedelta(seconds=i), 80.0 + i * 0.1) for i in range(10))
        self.assertGreater(risk.slope_per_second(series), 0)

    def test_slope_of_flat_series_is_zero(self):
        t0 = datetime.now()
        series = deque((t0 + timedelta(seconds=i), 80.0) for i in range(10))
        self.assertAlmostEqual(risk.slope_per_second(series), 0.0, places=6)


# ---------------------------------------------------------------- policy tiers

class PolicyTest(Base):
    def test_mold_candidates_are_tier1(self):
        e = self._engine()
        e.sensors.apply_scenario("mold")
        snap = e.sensors.snapshot()
        r = risk.mold_risk(snap, e.sensors.history)
        cands = policy.candidates(r, snap)
        kinds = [c.kind for c in cands]
        self.assertIn("ventilate_washer", kinds)
        self.assertIn("dehumidifier_on", kinds)
        for c in cands:
            self.assertEqual(c.tier, policy.TIER1)

    def test_thermal_candidates_split_across_tiers(self):
        e = self._engine()
        e.sensors.apply_scenario("oven")
        snap = e.sensors.snapshot()
        r = risk.thermal_risk(snap, e.sensors.history)
        cands = {c.kind: c for c in policy.candidates(r, snap)}
        self.assertEqual(cands["ventilate_kitchen"].tier, policy.TIER1)
        self.assertEqual(cands["delay_dryer"].tier, policy.TIER2)
        self.assertIsNotNone(cands["delay_dryer"].job)  # the 23:00 restart job

    def test_unknown_action_defaults_to_tier2(self):
        self.assertEqual(policy.tier_of("invented_action"), policy.TIER2)


# ---------------------------------------------------------------- approvals

class ApprovalTest(Base):
    def test_confirm_returns_the_actions(self):
        q = ApprovalQueue()
        ap = q.propose(risk_kind="thermal", question="Delay the dryer?",
                       summary="delay dryer", actions=[{"device_id": "dryer"}])
        got = q.confirm(ap.id)
        self.assertIsNotNone(got)
        self.assertEqual(got.status, "confirmed")
        self.assertIsNone(q.confirm(ap.id))  # not pending any more

    def test_decline(self):
        q = ApprovalQueue()
        ap = q.propose(risk_kind="thermal", question="q", summary="s", actions=[])
        self.assertEqual(q.decline(ap.id).status, "declined")

    def test_expiry_executes_nothing(self):
        q = ApprovalQueue(ttl_seconds=1)
        ap = q.propose(risk_kind="thermal", question="q", summary="s",
                       actions=[{"device_id": "dryer"}], ttl_seconds=0.2)
        time.sleep(0.3)
        expired = q.expire_due()
        self.assertEqual([a.id for a in expired], [ap.id])
        self.assertEqual(ap.status, "expired")
        self.assertIsNone(q.confirm(ap.id))  # too late to say yes


# ---------------------------------------------------------------- engine, no LLM

class ProactiveNoLLMTest(Base):
    def test_mold_scenario_runs_tier1_only(self):
        e = self._engine()
        e.sensors.apply_scenario("mold")
        out = e.tick()
        applied = {}
        for dev, acts in self.adapter.applied:
            applied.setdefault(dev, []).extend(a["action"] for a in acts)
        self.assertIn("ventilate", applied.get("washer", []))
        self.assertIn("power", applied.get("dehumidifier", []))
        self.assertEqual(e.approvals.pending(), [])  # nothing to ask about
        self.assertEqual(out["events"][0]["type"], "intervened")

    def test_thermal_scenario_executes_tier1_and_asks_tier2(self):
        e = self._engine()
        e.sensors.apply_scenario("oven")
        e.tick()
        applied = {}
        for dev, acts in self.adapter.applied:
            applied.setdefault(dev, []).extend(a["action"] for a in acts)
        self.assertIn("power", applied.get("range_hood", []))   # hood ran
        self.assertNotIn("dryer", applied)                       # dryer untouched
        pending = e.approvals.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].risk_kind, "thermal")
        self.assertTrue(pending[0].question)

    def test_confirmed_approval_executes_and_schedules_restart(self):
        e = self._engine()
        e.sensors.apply_scenario("oven")
        e.tick()
        ap = e.approvals.pending()[0]
        res = e.confirm(ap.id)
        self.assertTrue(res["ok"])
        applied = {}
        for dev, acts in self.adapter.applied:
            applied.setdefault(dev, []).extend(a["action"] for a in acts)
        self.assertIn("dryer", applied)  # now it ran
        jobs = [j for j in db.list_jobs() if j["device_id"] == "dryer"]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["rule"]["type"], "once")

    def test_declined_approval_changes_nothing(self):
        e = self._engine()
        e.sensors.apply_scenario("oven")
        e.tick()
        before = len(self.adapter.applied)
        ap = e.approvals.pending()[0]
        self.assertTrue(e.decline(ap.id)["ok"])
        self.assertEqual(len(self.adapter.applied), before)
        self.assertEqual(e.approvals.pending(), [])

    def test_expired_approval_changes_nothing(self):
        e = self._engine()
        e.approvals.ttl_seconds = 0.2
        e.sensors.apply_scenario("oven")
        e.tick()
        ap = e.approvals.pending()[0]
        time.sleep(0.3)
        e.tick()  # expiry pass
        self.assertEqual(e.approvals.get(ap.id).status, "expired")
        applied_devs = {dev for dev, _ in self.adapter.applied}
        self.assertNotIn("dryer", applied_devs)

    def test_second_tick_does_not_duplicate_a_pending_ask(self):
        e = self._engine()
        e.sensors.apply_scenario("oven")
        e.tick()
        e.tick()
        self.assertEqual(len(e.approvals.pending()), 1)

    def test_cooldown_blocks_immediate_repeat(self):
        e = self._engine(_cfg(cooldown_seconds=600))
        e.sensors.apply_scenario("mold")
        e.tick()
        n = len(self.adapter.applied)
        e.tick()
        self.assertEqual(len(self.adapter.applied), n)


# ---------------------------------------------------------------- engine, LLM decides

class ProactiveLLMTest(Base):
    def _engine_llm(self, fake, **proactive):
        cfg = _cfg(llm={"enabled": True, "base_url": fake.base_url, "api_key": "k",
                        "model": "m", "timeout": 5}, **proactive)
        return self._engine(cfg)

    def test_llm_subset_is_respected(self):
        with FakeLLM() as fake:
            fake.prose_reply = ('{"chosen": [{"kind": "ventilate_washer", '
                                '"reason": "drum is wet"}], "note": "drum drying"}')
            e = self._engine_llm(fake)
            e.sensors.apply_scenario("mold")
            e.tick()
            applied = {}
            for dev, acts in self.adapter.applied:
                applied.setdefault(dev, []).extend(a["action"] for a in acts)
            self.assertIn("ventilate", applied.get("washer", []))
            self.assertNotIn("dehumidifier", applied)  # the model left it alone

    def test_llm_abstention_does_nothing(self):
        with FakeLLM() as fake:
            fake.prose_reply = ('{"chosen": [], "note": "trend is flat, drum is drying"}')
            e = self._engine_llm(fake)
            e.sensors.apply_scenario("mold")
            e.tick()
            self.assertEqual(self.adapter.applied, [])
            self.assertEqual(e.approvals.pending(), [])

    def test_llm_garbage_falls_back_to_rule_default(self):
        with FakeLLM() as fake:
            fake.prose_reply = "I am not JSON"
            e = self._engine_llm(fake)
            e.sensors.apply_scenario("mold")
            e.tick()
            applied = {}
            for dev, acts in self.adapter.applied:
                applied.setdefault(dev, []).extend(a["action"] for a in acts)
            self.assertIn("ventilate", applied.get("washer", []))
            self.assertIn("power", applied.get("dehumidifier", []))

    def test_llm_choosing_only_tier1_skips_the_question(self):
        with FakeLLM() as fake:
            fake.prose_reply = '{"chosen": [{"kind": "ventilate_kitchen"}], "note": "exhaust the heat"}'
            e = self._engine_llm(fake)
            e.sensors.apply_scenario("oven")
            e.tick()
            applied = {dev for dev, _ in self.adapter.applied}
            self.assertIn("range_hood", applied)
            self.assertNotIn("dryer", applied)
            self.assertEqual(e.approvals.pending(), [])  # delay was never chosen


if __name__ == "__main__":
    unittest.main()
