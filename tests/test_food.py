"""Food storage guidance.

The interesting property here is not that it answers — it is that it answers
from two sources at once: a rule from outside and a temperature read from the
appliance. A model has the first and not the second, which is why the model is
not consulted and why these tests assert on the number in the reply.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, db  # noqa: E402
from app import food  # noqa: E402
from app import nlu  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.devices import MockAdapter  # noqa: E402
from app.devices.base import CAPABILITIES  # noqa: E402
from app.devices import mock as mock_mod  # noqa: E402

# Built from the table, not from `MockAdapter().list_devices()`: constructing an
# adapter at import time starts a thread that persists every five seconds, and
# on the very first import it reads the real data/ directory.
DEVICES = [
    {**d,
     "online": True,
     "capabilities": CAPABILITIES.get(d["type"], ["power"]),
     "type_label": mock_mod.TYPE_LABELS.get(d["type"], d["type"])}
    for d in mock_mod.DEVICES
]


class MatchTest(unittest.TestCase):
    def test_finds_the_food_in_a_long_sentence(self):
        self.assertEqual(food.match_food("can I keep the eggs in the fridge")["label"],
                         "eggs")

    def test_no_food_word_means_no_answer(self):
        self.assertIsNone(food.match_food("turn the fridge down to 3"))
        self.assertFalse(food.is_food_request("turn the fridge down to 3"))

    def test_food_without_a_storage_question_is_not_answered_here(self):
        # "what can I cook with eggs" mentions food but asks nothing this
        # module can answer from the house; it belongs to the refusal path.
        self.assertIsNotNone(food.match_food("what can I cook with eggs"))
        self.assertFalse(food.is_food_request("what can I cook with eggs"))


class ParseOrderTest(unittest.TestCase):
    """The comfort rules would heat the fridge, so this has to come first."""

    def test_cold_enough_is_not_a_complaint_about_the_cold(self):
        it = nlu.parse("is the fridge cold enough for vegetables", DEVICES,
                       None, "Living Room")
        self.assertEqual(it["intent"], "food")
        # measured on this build before the fix: power on, mode heat, 24 °C
        self.assertNotIn("heat", [a.get("value") for a in it.get("actions", [])])

    def test_target_is_the_refrigerator(self):
        it = nlu.parse("where should I store the milk", DEVICES, None, "Living Room")
        self.assertEqual(it["intent"], "food")
        self.assertEqual(it["device_id"], "fridge")

    def test_a_temperature_command_is_not_hijacked(self):
        # Not asserted as "control": a refrigerator setpoint is 1–7 °C and the
        # action extractor only accepts 16–30, so this sentence lands elsewhere
        # for reasons that predate this module. What matters is that it is not
        # mistaken for a question about food.
        it = nlu.parse("set the fridge to 3 degrees", DEVICES, None, "Kitchen")
        self.assertNotEqual(it["intent"], "food")
        self.assertFalse(it.get("food", False))

    def test_recipes_are_refused_not_invented(self):
        it = nlu.parse("recommend a recipe for dinner", DEVICES, None, "Living Room")
        self.assertTrue(it.get("out_of_scope"))
        self.assertIn("recipe", it["clarify"].lower())


class ReplyTest(unittest.TestCase):
    def setUp(self):
        # The agent writes every turn to SQLite, so this needs the same
        # isolation the other agent tests use. Without it the suite reads and
        # writes the real data/ directory, which works on a machine that has
        # already run the app and fails on a fresh one.
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._orig_data_dir = db.DATA_DIR
        self._orig_db_path = db.DB_PATH
        self._orig_state_path = mock_mod.STATE_PATH
        db.DATA_DIR = self.tmp
        db.DB_PATH = self.tmp / "test.db"
        mock_mod.STATE_PATH = self.tmp / "mock_state.json"
        db.init()
        self.adapter = MockAdapter()

    def tearDown(self):
        self.adapter.stop()
        db.DATA_DIR = self._orig_data_dir
        db.DB_PATH = self._orig_db_path
        mock_mod.STATE_PATH = self._orig_state_path
        self._tmp.cleanup()

    def fridge(self):
        d = self.adapter.get_device("fridge")
        return {**d, "state": self.adapter.get_state("fridge")}

    def test_both_compartments_appear_with_the_measured_temperature(self):
        reply = food.answer("should I put the chicken in the fridge or the freezer",
                            self.fridge())
        # the model's answer to this was "either", which is not an answer
        self.assertNotIn("either", reply.lower())
        self.assertIn("3 °C", reply)
        self.assertIn("-18", reply)
        self.assertIn("one to two days", reply)
        self.assertIn("several months", reply)

    def test_a_named_temperature_is_judged_not_replaced(self):
        reply = food.answer("is 4 degrees cold enough for the chicken",
                            self.fridge())
        self.assertIn("4 °C is cold enough", reply)
        self.assertIn("Right now the compartment is at 3 °C", reply)

    def test_no_refrigerator_says_so_instead_of_inventing_one(self):
        reply = food.answer("where should I store the milk", None)
        self.assertIn("I cannot read a refrigerator", reply)
        self.assertNotIn("°C", reply)

    def test_a_typo_still_reaches_the_table(self):
        cfg = dict(config.DEFAULTS)
        cfg["agent"] = {"default_room": "Living Room"}
        agent = Agent(self.adapter, cfg)
        result = agent.handle("shud i put the chiken in the fridge")
        self.assertEqual(result.get("intent_name"), "food")
        self.assertIn("chicken", result["reply"].lower())


if __name__ == "__main__":
    unittest.main()
