"""Tests for the scene layer: named routines, resolved without a model.

These exist because the benchmark showed two sentences that looked like model
failures and were really a missing tier:

    wake the house up                        -> unknown, then 11 s and no action
    put the whole flat to sleep              -> 7 steps, 15 s, sometimes a claim
    put the living room to sleep in 20 min   -> acted immediately, or lied

All three are routines. A routine is a table, so all three are now microseconds
and cannot be wrong in the way a model can.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, db, scenes  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.devices import MockAdapter  # noqa: E402
from app.devices import mock as mock_mod  # noqa: E402


class SceneTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._orig = (db.DATA_DIR, db.DB_PATH, mock_mod.STATE_PATH)
        db.DATA_DIR = self.tmp
        db.DB_PATH = self.tmp / "test.db"
        mock_mod.STATE_PATH = self.tmp / "mock_state.json"
        db.init()

        self.adapter = MockAdapter()
        cfg = dict(config.DEFAULTS)
        cfg["agent"] = {"default_room": "Living Room"}
        cfg["llm"] = {**cfg.get("llm", {}), "enabled": False}
        self.cfg = cfg
        self.agent = Agent(self.adapter, cfg)

    def tearDown(self):
        self.adapter.stop()
        db.DATA_DIR, db.DB_PATH, mock_mod.STATE_PATH = self._orig
        self._tmp.cleanup()

    def state(self, device_id: str) -> dict:
        return self.adapter.get_state(device_id)


class MorningTest(SceneTestBase):
    def test_wake_the_house_up_turns_on_the_rooms(self):
        r = self.agent.handle("wake the house up")

        self.assertEqual(r["intent_name"], "scene")
        self.assertTrue(r["ok"])
        self.assertTrue(self.state("ac_living")["power"])
        self.assertTrue(self.state("ac_bedroom")["power"])
        self.assertTrue(self.state("purifier")["power"])

    def test_the_washer_is_not_part_of_waking_up(self):
        """"Waking the house" is not "switching on every appliance"."""
        before = {d: self.state(d)["power"] for d in ("washer", "fridge")}
        self.agent.handle("wake the house up")

        for device_id, was in before.items():
            self.assertEqual(self.state(device_id)["power"], was, device_id)

    def test_good_morning_is_the_same_routine(self):
        r = self.agent.handle("good morning")
        self.assertEqual(r["intent_name"], "scene")
        self.assertTrue(self.state("ac_bedroom")["power"])

    def test_no_model_is_consulted(self):
        """The point of the layer: this cannot fail the way a model can."""
        with mock.patch("app.llm.chat") as chat:
            self.agent.handle("wake the house up")
            chat.assert_not_called()


class NightTest(SceneTestBase):
    def test_put_the_whole_flat_to_sleep(self):
        r = self.agent.handle("put the whole flat to sleep")

        self.assertEqual(r["intent_name"], "scene")
        for device_id in ("ac_living", "ac_bedroom", "purifier", "washer"):
            self.assertFalse(self.state(device_id)["power"], device_id)

    def test_the_fridge_stays_on(self):
        """The one judgement a routine makes that no sentence states."""
        self.adapter.apply("fridge", [{"action": "power", "value": True}])
        r = self.agent.handle("good night")

        self.assertTrue(self.state("fridge")["power"])
        self.assertIn("Fridge stays on", r["reply"])

    def test_get_ready_for_bed(self):
        self.agent.handle("get the flat ready for bed")
        self.assertFalse(self.state("ac_living")["power"])

    def test_leaving_is_the_away_routine(self):
        r = self.agent.handle("I'm leaving")
        self.assertEqual(r["intent_name"], "scene")
        self.assertFalse(self.state("ac_living")["power"])
        self.assertFalse(self.state("washer")["power"])


class ScopeTest(SceneTestBase):
    def test_a_room_limits_the_routine(self):
        self.adapter.apply("ac_living", [{"action": "power", "value": True}])
        self.adapter.apply("ac_bedroom", [{"action": "power", "value": True}])
        self.agent.handle("put the bedroom to sleep")

        self.assertFalse(self.state("ac_bedroom")["power"])
        self.assertTrue(self.state("ac_living")["power"],
                        "a bedroom routine must not touch the living room")

    def test_a_room_with_a_time_schedules_only_that_room(self):
        self.agent.handle("put the living room to sleep in 20 minutes")
        jobs = db.list_jobs()

        self.assertEqual(sorted(j["device_id"] for j in jobs),
                         ["ac_living", "purifier"])
        self.assertTrue(all(j["rule"]["type"] == "once" for j in jobs))


class NotASceneTest(SceneTestBase):
    def test_an_explicit_command_is_not_hijacked(self):
        r = self.agent.handle("turn off everything in the living room")
        self.assertEqual(r["intent_name"], "control")

    def test_a_plain_device_command_stays_a_device_command(self):
        r = self.agent.handle("turn on the bedroom AC")
        self.assertEqual(r["intent_name"], "control")

    def test_an_unrelated_sentence_is_not_a_routine(self):
        self.assertIsNone(scenes.detect("the kitchen smells"))
        self.assertIsNone(scenes.detect("turn the AC to 24"))


class KeepRunningTest(unittest.TestCase):
    """Every off-scene must spare the fridge. Written as a property because the
    next scene someone adds is the one that will forget."""

    def test_no_off_scene_touches_keep_running_types(self):
        fridge = {"id": "fridge", "name": "Kitchen Fridge", "type": "refrigerator"}
        for key, spec in scenes.SCENES.items():
            if not spec.get("default"):
                continue
            if spec["default"][0].get("value") is False:
                self.assertIsNone(scenes.actions_for(key, fridge), key)

    def test_the_table_is_readable(self):
        """If a routine needs a comment to explain, it needs a test instead."""
        for key, spec in scenes.SCENES.items():
            self.assertTrue(spec.get("label"), key)
            for dtype, acts in spec["by_type"].items():
                for a in acts:
                    self.assertIn(a["action"], ("power", "temperature", "mode", "fan"))


if __name__ == "__main__":
    unittest.main()
