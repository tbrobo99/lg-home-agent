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

    def test_a_bare_ingredient_is_a_storage_question(self):
        # Typed into the deployed build, "chicken" got the generic help text,
        # which reads as "I did not understand" when the word was clear.
        for text in ("chicken", "the chicken", "and chicken?", "what about the milk"):
            self.assertTrue(food.is_food_request(text), text)

    def test_a_bare_ingredient_with_a_recipe_verb_still_is_not(self):
        self.assertFalse(food.is_food_request("what can I cook with eggs and tomatoes"))

    def test_a_device_is_not_an_ingredient(self):
        self.assertFalse(food.is_food_request("fridge"))
        self.assertFalse(food.is_food_request("turn on the living room AC"))


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

    def test_the_word_the_user_typed_is_the_word_in_the_answer(self):
        """The label is a category; the user asked about one food.

        Answering "apple" with "Most fruit keep three to seven days" is an
        answer to a question nobody asked.
        """
        reply = food.answer("apple", self.fridge())
        self.assertIn("Apple goes in the fridge", reply)
        self.assertNotIn("Most fruit", reply)

    def test_a_qualifier_in_the_label_is_kept(self):
        """ "Rice" must not be answered as if it were cooked rice."""
        self.assertIn("Cooked rice goes in the fridge", food.answer("rice", self.fridge()))
        self.assertIn("Raw chicken goes in the fridge if",
                      food.answer("chicken", self.fridge()))

    def test_a_narrowed_label_says_which_one_it_means(self):
        """The table narrowed "rice" to cooked rice; the user did not.

        Answering "Cooked rice keeps about a day" to somebody holding a bag of
        basmati is answering a neighbouring question, so the reply says which
        one it is about — and stays quiet when the user was already specific.
        """
        bare = food.answer("rice", self.fridge())
        self.assertIn("Dry rice belongs in the cupboard", bare)

        explicit = food.answer("cooked rice", self.fridge())
        self.assertNotIn("Dry rice", explicit)

    def test_a_singular_food_does_not_get_a_plural_pronoun(self):
        """ "Apple … they freeze well" — the label's pronoun, on a singular word.

        Every phrase in the table has to read correctly under any subject,
        because the subject is replaced by the word the user typed.
        """
        for text in ("apple", "spinach", "banana"):
            reply = food.answer(text, self.fridge())
            self.assertNotIn(" they ", reply, text)
            self.assertNotIn(" it ", reply, text)

    def test_an_entry_without_a_duration_is_not_dropped_into_a_duration_sentence(self):
        """ "Ice cream keeps it does not belong there, so 3 °C is cold enough." """
        reply = food.answer("is the fridge cold enough for ice cream", self.fridge())
        self.assertNotIn("keeps it does not belong", reply)
        self.assertIn("does not belong in the fridge", reply)
        self.assertIn("goes in the freezer, always", reply)

    def test_a_verdict_is_a_phrase_not_a_comparison_with_nothing(self):
        """ "8 °C is warmer than that likes" had no antecedent for "that"."""
        warm = food.answer("is 8 degrees enough for chicken", self.fridge())
        self.assertIn("8 °C is too warm", warm)
        self.assertNotIn("than that likes", warm)

        mild = food.answer("is 5 degrees enough for the milk", self.fridge())
        self.assertIn("5 °C is on the warm side", mild)

    def test_the_appliance_can_be_judged_with_no_ingredient_named(self):
        """ "is the fridge cold enough" used to fall through to the help text."""
        self.assertTrue(food.is_food_request("is the fridge cold enough"))
        fridge = food.answer("is the fridge cold enough", self.fridge())
        self.assertIn("0-4 °C", fridge)
        self.assertIn("3 °C is cold enough", fridge)

        freezer = food.answer("is the freezer cold enough", self.fridge())
        self.assertIn("-18", freezer)

    def test_a_complaint_about_the_fridge_is_not_a_question_about_its_number(self):
        """ "the fridge is too warm" wants it colder, not a reading of 3 °C.

        Only the question form is answered here. The statement belongs to the
        layers that can act on it.
        """
        self.assertFalse(food.is_food_request("the fridge is too warm"))
        self.assertTrue(food.is_food_request("is the fridge too warm?"))

    def test_cured_meat_is_not_raw_meat(self):
        """Bacon was in the raw-meat entry, which said one to two days."""
        reply = food.answer("bacon", self.fridge())
        self.assertIn("a week or two", reply)
        self.assertNotIn("one to two days", reply)

    def test_a_bare_ingredient_is_told_which_compartment(self):
        """ "Apple keeps three to seven days · freezer: good frozen too" answers
        neither question. Somebody holding an apple wants the compartment."""
        reply = food.answer("apple", self.fridge())
        self.assertIn("Apple goes in the fridge", reply)

        # the other way round for fish, which is not improved by a day in the fridge
        self.assertIn("Raw fish goes in the freezer", food.answer("fish", self.fridge()))

    def test_raw_meat_is_a_condition_rather_than_a_compartment(self):
        """When it goes in the fridge depends on when it will be cooked, and
        saying so is more useful than picking one and being wrong."""
        reply = food.answer("should I put the chicken in the fridge or the freezer",
                            self.fridge())
        self.assertIn("the fridge if you will use it within one to two days", reply)
        self.assertIn("otherwise the freezer, where it keeps several months", reply)

    def test_the_longest_key_wins(self):
        """ "Ice cream" contains "cream", and cream belongs to milk."""
        self.assertIn("Ice cream", food.answer("ice cream", self.fridge()))
        self.assertNotIn("printed date", food.answer("ice cream", self.fridge()))

    def test_the_source_note_is_said_once_not_every_time(self):
        cfg = dict(config.DEFAULTS)
        cfg["agent"] = {"default_room": "Living Room"}
        agent = Agent(self.adapter, cfg)

        first = agent.handle("chicken")["reply"]
        second = agent.handle("apple")["reply"]

        self.assertIn("general guidance", first)
        self.assertNotIn("general guidance", second)

    def test_the_answer_is_short_lines_not_a_paragraph(self):
        """Three ingredients in a row used to be three identical paragraphs."""
        reply = food.answer("fish", self.fridge(), note=False)
        lines = reply.split("\n")
        self.assertEqual(len(lines), 3)
        self.assertLess(max(len(line) for line in lines), 60)

    def test_no_line_starts_with_a_stranded_separator(self):
        """On a phone the bubble wraps, and "· freezer: …" began with a dot."""
        for text in ("apple", "cheese", "spinach", "rice"):
            for line in food.answer(text, self.fridge()).split("\n"):
                self.assertFalse(line.startswith("·"), f"{text}: {line!r}")


if __name__ == "__main__":
    unittest.main()
