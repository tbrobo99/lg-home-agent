"""Food storage guidance, paired with the appliance that is actually there.

A recipe recommendation is a party trick. Any model can produce one, nothing in
it depends on this house, and nobody can check it. Storage advice is different:
the answer is a number read from the appliance plus a rule from outside, which is
the one place this project can say something a chatbot cannot.

The guidance below is ordinary food-safety advice, not a standard. It is worded
as ranges for that reason, and the reply says where it comes from.

Every phrase in the table is written to survive being dropped into a sentence
with *any* subject, singular or plural: no "it", no "they". The first draft had
"they freeze well" for fruit, and "apple" — which is in the table as *most
fruit* — came back as "Apple … freezer: they freeze well". A table of phrases
that only read correctly under the label they were written for is a table that
produces ungrammatical answers the moment the label is replaced by the word the
user typed, and the label is replaced on purpose.
"""
from __future__ import annotations

import re

# keys -> what to call it, which compartment to reach for, how long it keeps in
# each, and an optional aside that qualifies the first number.
#
# `pick` is the part that answers the question people actually ask. "Apple keeps
# three to seven days · freezer: good frozen too" is a table, not advice: the
# user has an apple in their hand and wants to be told where to put it. So each
# entry names the compartment it belongs in — or says the choice depends on when
# you will cook it, which is the honest answer for raw meat.
FOOD_TABLE: list[dict] = [
    {"keys": ["chicken", "hen", "poultry"],
     "label": "raw chicken",
     "pick": "depends",
     "keeps": "one to two days",
     "frozen": "several months"},
    {"keys": ["beef", "pork", "lamb", "meat", "mince", "minced beef",
              "ground beef"],
     "label": "raw meat",
     "pick": "depends",
     "keeps": "one to two days",
     "frozen": "several months"},
    # Cured meat was in the raw-meat entry, which told anyone asking about bacon
    # that it keeps one to two days. It keeps for weeks; the date on the packet
    # is the number that matters.
    {"keys": ["bacon", "sausage", "sausages", "ham", "salami", "chorizo"],
     "label": "cured meat",
     "pick": "fridge",
     "keeps": "a week or two once opened",
     "aside": "the printed date decides",
     "frozen": "freezes fine, but there is no need"},
    {"keys": ["fish", "salmon", "tuna", "cod", "prawn", "prawns", "shrimp",
              "seafood", "crab"],
     "label": "raw fish",
     "pick": "freezer",
     "keeps": "about a day",
     "frozen": "several months"},
    {"keys": ["milk", "cream"],
     "label": "milk",
     "pick": "fridge",
     "keeps": "a few days",
     "aside": "the printed date decides",
     "frozen": "can be frozen but separates when thawed"},
    {"keys": ["egg", "eggs"],
     "label": "eggs",
     "pick": "fridge",
     "keeps": "three to five weeks",
     "frozen": "only out of the shell"},
    {"keys": ["cooked rice", "leftover rice", "boiled rice", "rice"],
     "label": "cooked rice",
     "pick": "fridge",
     "keeps": "about a day",
     "aside": "reheat only once",
     "frozen": "a month or more",
     # The label narrows a word the user did not narrow. Saying so is the
     # difference between answering a question and answering a neighbouring one.
     "if_matched": {"rice": "Dry rice belongs in the cupboard, not the fridge."}},
    {"keys": ["leftovers", "leftover", "takeaway", "takeout"],
     "label": "leftovers",
     "pick": "fridge",
     "keeps": "one to two days",
     "frozen": "a month or more"},
    {"keys": ["butter"],
     "label": "butter",
     "pick": "fridge",
     "keeps": "a few weeks once opened",
     "frozen": "freezes well"},
    {"keys": ["cheese"],
     "label": "cheese",
     "pick": "fridge",
     "keeps": "a few weeks once opened",
     "aside": "a hard cheese keeps longer",
     "frozen": "can be frozen, but crumbles when thawed"},
    {"keys": ["vegetable", "vegetables", "veggies", "lettuce", "spinach",
              "carrot", "carrots", "broccoli", "cabbage", "cucumber", "salad",
              "greens", "herb", "herbs", "mushroom", "mushrooms", "onion",
              "onions", "potato", "potatoes"],
     "label": "most vegetables",
     "pick": "fridge",
     "keeps": "three to seven days",
     "aside": "away from the coldest part",
     "frozen": "blanch first"},
    {"keys": ["fruit", "fruits", "berries", "strawberry", "strawberries",
              "banana", "bananas", "apple", "apples", "orange", "oranges",
              "grape", "grapes"],
     "label": "most fruit",
     "pick": "fridge",
     "keeps": "three to seven days",
     "frozen": "good frozen too"},
    # No duration, so this entry is a sentence rather than a phrase: there is no
    # number of days it keeps in a fridge, because it is not supposed to be in
    # one. The first draft gave it "keeps": "it does not belong there", and the
    # adequacy path produced "Ice cream keeps it does not belong there, so 3 °C
    # is cold enough for that."
    {"keys": ["ice cream", "icecream"],
     "label": "ice cream",
     "keeps": None,
     "line": "goes in the freezer, always: it does not belong in the fridge at all",
     "fresh": "does not belong in the fridge at all",
     "frozen": "the freezer is the only place for it"},
]

# "put the chicken in the fridge" — a place is named, so the answer compares them.
STORAGE_INTENT_RE = re.compile(
    r"\b(?:keep|kept|storing|store|stored|put|place|leave|saving|save|"
    r"refrigerate|chill|freeze|frozen|freezer|fresh)\b"
    r"|\bwhere\s+(?:should|do|to|does)\b"
    r"|\bhow\s+long\b", re.I)

# "is the fridge cold enough" — no place named, so the answer is a judgement
# about the temperature that is measured right now.
ADEQUACY_RE = re.compile(
    r"\b(?:enough|adequate|okay|ok|safe|good|alright)\b"
    r"|\btoo\s+(?:warm|hot|cold)\b"
    r"|\bis\s+it\s+(?:cold|warm|fine)\b", re.I)

# The appliance itself, for the question that names no ingredient at all.
FRIDGE_RE = re.compile(r"\b(?:fridge|refrigerator|freezer|compartment)\b", re.I)


def match_food(text: str) -> dict | None:
    """Return the entry whose trigger word best matches, longest word first.

    Longest first, because the keys overlap: "ice cream" contains "cream", and
    taking the first entry that matched in table order answered "ice cream" with
    the milk rule. The entry also carries the key that matched, because the label
    is a category and the user typed a food — "apple" is in the table as *most
    fruit*, and telling somebody who asked about an apple that "most fruit" keep
    three to seven days is an answer to a question they did not ask.
    """
    t = (text or "").lower()
    best, best_len = None, 0
    for entry in FOOD_TABLE:
        for key in entry["keys"]:
            if len(key) <= best_len:
                continue
            if re.search(r"\b" + re.escape(key) + r"\b", t):
                best, best_len = {**entry, "matched": key}, len(key)
    return best


def wants_storage(text: str) -> bool:
    return bool(STORAGE_INTENT_RE.search(text or ""))


def wants_adequacy(text: str) -> bool:
    return bool(ADEQUACY_RE.search(text or ""))


def is_fridge_adequacy(text: str) -> bool:
    """ "is the fridge cold enough" — a judgement about the appliance itself.

    No ingredient is named, so there is no rule to quote; the whole answer is
    the reading and whether it is in range. This used to fall out of the food
    module entirely and land on the generic help text, which is the wrong answer
    to a question this house can answer exactly.

    "No ingredient" is the test, not "no storage word". "freezer" is a storage
    word, so excluding storage questions outright dropped "is the freezer cold
    enough" — which is the same question about the other compartment.
    """
    if not wants_adequacy(text) or not FRIDGE_RE.search(text or ""):
        return False
    if match_food(text) is not None:
        return False
    # A question, not a complaint. "the fridge is too warm" is a statement about
    # a fault — the answer to it is to make the fridge colder, not to read the
    # number back — and it belongs to the layers that can act. "is the fridge
    # cold enough" is a question, and the reading is the whole answer.
    t = (text or "").strip()
    return bool(t.endswith("?") or re.match(r"(?:is|are|does|do|should|can|"
                                             r"how)\b", t, re.I))


# Words that can sit around an ingredient without making the sentence about
# something else: "chicken", "the chicken", "and chicken?".
_FILLER = frozenset("""a an the some this that it and or for me my about what
whats how long is are do does can i should put keep store of""".split())


def is_food_request(text: str) -> bool:
    """True when the sentence is about storing food rather than about a device.

    Guarded on purpose. "turn the fridge down to 3" mentions no food, so it
    stays a control command; "can I keep the chicken in the fridge" names food
    and a place, so it never reaches the comfort rules, which would read the
    word "cold" as a request to heat the room.

    A bare ingredient counts as a storage question. Measured on the deployed
    build: typing "chicken" got the generic help text, which reads as "I did not
    understand" when the word was perfectly clear and the only thing this house
    can say about an ingredient is how long it keeps.
    """
    if not text:
        return False
    entry = match_food(text)
    if entry is None:
        return is_fridge_adequacy(text)
    if wants_storage(text) or wants_adequacy(text):
        return True
    # Count what is left after the food itself. Counting words was enough for
    # "chicken" and wrong for "ice cream": two words, so the bare-ingredient rule
    # did not fire, and the phrase fell through to the model, which answered
    # "I cannot assist with that request."
    rest = re.sub(r"\b" + re.escape(entry.get("matched") or "") + r"\b", " ",
                  text.lower())
    words = [w for w in re.findall(r"[A-Za-z']+", rest) if w not in _FILLER]
    return len(words) <= 1


# "what temperature is that" asked with a number in it, e.g. "is 4 degrees
# enough". The user is asking about a value, so answer about that value.
DEGREES_RE = re.compile(r"\b(\d{1,2})\s*(?:°\s*c|°|degrees?(?:\s+celsius)?|"
                        r"celsius)\b", re.I)


def _subject(food: dict) -> str:
    """What to call the thing, given what the user called it.

    The label wins when it qualifies the food in a way the word itself does not
    — "rice" is *cooked rice* in this table, and answering "Rice keeps about a
    day" would be wrong about a bag of basmati. Everywhere else the matched word
    wins, because the label is a category: "Most fruit" is an answer to a
    question nobody asked when the word was "apple", and "Butter and cheese" is
    a worse answer to "butter" than "Butter" is.
    """
    label, matched = food["label"], food.get("matched") or food["label"]
    qualifier = label.split(" ", 1)[0].lower()
    # An entry with no duration is written as a full sentence, and a sentence
    # needs the noun phrase rather than whatever spelling the user happened to
    # type ("icecream").
    if food.get("keeps") is None:
        subject = label
    else:
        subject = label if qualifier in ("raw", "cooked") else matched
    return subject[:1].upper() + subject[1:]


def _verb(subject: str, singular: str = "keeps", plural: str = "keep") -> str:
    """keeps / keep — "Raw chicken keeps", "Vegetables keep"."""
    return plural if subject.lower().endswith("s") else singular


def _guidance(food: dict, subject: str) -> list[str]:
    """Where to put it, then how long it lasts there.

    Order matters, and so does the verb. "Apple keeps three to seven days ·
    freezer: good frozen too" lists two numbers and answers neither question:
    somebody holding an apple wants to be told which compartment, and a reply
    that names both is a table read aloud. So the first line says where the food
    goes — *goes in the fridge*, not "fridge:" — and the other compartment
    follows as the alternative it is.

    Raw meat is the case where the honest answer is a condition rather than a
    compartment: the fridge if you will cook it soon, the freezer otherwise, so
    `pick: "depends"` says that instead of guessing when you eat.
    """
    if food.get("line"):
        return [f"{subject} {food['line']}."]

    keeps, frozen = food["keeps"], food["frozen"]
    pick = food.get("pick", "fridge")

    if pick == "freezer":
        return [f"{subject} {_verb(subject, 'goes', 'go')} in the freezer — {frozen}",
                f"Fridge: {keeps}"]

    if pick == "depends":
        return [f"{subject} {_verb(subject, 'goes', 'go')} in the fridge if you "
                f"will use it within {keeps}; otherwise the freezer, where it "
                f"keeps {frozen}."]

    first = f"{subject} {_verb(subject, 'goes', 'go')} in the fridge — {keeps}"
    if food.get("aside"):
        first += f" ({food['aside']})"
    return [first, f"Freezer: {frozen}"]


def _note() -> str:
    """Where the guidance comes from. Short enough to survive being repeated."""
    return "(general guidance, not a reading from the fridge)"


def _verdict(temp: int | None, freezer: bool = False) -> str:
    """A phrase, not a comparison. "so 8 °C is warmer than that likes" had no
    antecedent for "that" — it was a template talking to itself."""
    if temp is None:
        return ""
    if freezer:
        return "fine" if temp <= -18 else "not cold enough"
    if temp <= 4:
        return "cold enough"
    if temp <= 7:
        return "on the warm side"
    return "too warm"


def _reading(name: str, fresh: int | None, frozen: int | None) -> str:
    if frozen is not None:
        return f"{name}: {fresh} °C · freezer {frozen} °C"
    return f"{name}: {fresh} °C"


def _appliance_answer(text: str, name: str, fresh, frozen, note: bool) -> str:
    """No ingredient named: judge the appliance against the usual range."""
    freezer = bool(re.search(r"\bfreezer\b", text or "", re.I))
    lines = [_reading(name, fresh, frozen)] if fresh is not None else []
    if freezer and frozen is not None:
        lines.append(f"A freezer wants -18 °C or colder, so {frozen} °C is "
                     f"{_verdict(frozen, freezer=True)}.")
    elif fresh is not None:
        lines.append(f"A fridge wants 0-4 °C, so {fresh} °C is "
                     f"{_verdict(fresh)}.")
    if note and lines:
        lines.append(_note())
    return "\n".join(lines)


def answer(text: str, fridge: dict | None, note: bool = True) -> str:
    """Compose the reply: the measured temperature, the guidance, its source.

    Three short lines rather than a paragraph. Measured on the deployed build,
    three storage questions in a row produced three identical
    "Kitchen Fridge is running at 3 °C with a freezer at -18 °C." sentences
    followed by three identical fourteen-word disclaimers, which is a wall of
    text with two useful facts in it.

    `fridge` is the device record plus its live state, or None when this house
    has no refrigerator to read. The guidance still applies either way; only the
    numbers from the house are missing, and the reply says so instead of
    inventing a temperature.
    """
    name, fresh_temp, frozen_temp = "", None, None
    if fridge:
        name = fridge.get("name", "the refrigerator")
        fresh_temp = (fridge.get("state") or {}).get("fridge_temp")
        frozen_temp = (fridge.get("state") or {}).get("freezer_temp")

    food = match_food(text)
    if food is None:
        # "is the fridge cold enough" — nothing to look up, so the answer is the
        # reading and the range it is being judged against.
        if not is_fridge_adequacy(text):
            return ""
        if not name or fresh_temp is None:
            return ("This house has no refrigerator I can read, so I cannot "
                    "say whether it is cold enough.")
        return _appliance_answer(text, name, fresh_temp, frozen_temp, note)

    lines: list[str] = []
    if name and fresh_temp is not None:
        lines.append(_reading(name, fresh_temp, frozen_temp))

    subject = _subject(food)
    keeps = food.get("keeps")

    if keeps is None:
        # An entry with no duration is stored as a sentence, and the sentence
        # already says which compartment it belongs in.
        lines.extend(_guidance(food, subject))
    elif wants_adequacy(text) and not wants_storage(text):
        # A judgement, not a comparison. If the user named a temperature, judge
        # that one and say what the house is actually at; otherwise judge the
        # reading. Answering the measured number at a question that asked about
        # 4 °C is the same failure as answering a different question entirely.
        m = DEGREES_RE.search(text or "")
        asked = int(m.group(1)) if m else None
        judge = fresh_temp
        source = f"{fresh_temp} °C"
        if asked is not None and 0 <= asked <= 30:
            judge, source = asked, f"{asked} °C"
        elif fresh_temp is None:
            lines.append("This house has no refrigerator I can read, so I can "
                         "only judge the guidance, not your appliance.")
            lines.append(f"{subject}: {food['fresh']} in a fresh-food "
                         f"compartment.")
            return "\n".join(lines)
        sentence = f"{subject} {_verb(subject)} {keeps}"
        if food.get("aside"):
            sentence += f" — {food['aside']}"
        lines.append(f"{sentence}. {source} is {_verdict(judge)}.")
        if asked is not None and fresh_temp is not None and asked != fresh_temp:
            lines.append(f"Right now the compartment is at {fresh_temp} °C.")
        if note:
            lines.append(_note())
        return "\n".join(lines)
    else:
        lines.extend(_guidance(food, subject))

    # The label narrowed a word the user did not narrow; say which one is meant.
    narrowed = (food.get("if_matched") or {}).get(food.get("matched") or "")
    if narrowed:
        lines.append(narrowed)

    if not name:
        lines.append("I cannot read a refrigerator here, so this is the general "
                     "rule rather than your appliance.")
    elif note:
        lines.append(_note())
    return "\n".join(lines)
