"""Food storage guidance, paired with the appliance that is actually there.

A recipe recommendation is a party trick. Any model can produce one, nothing in
it depends on this house, and nobody can check it. Storage advice is different:
the answer is a number read from the appliance plus a rule from outside, which is
the one place this project can say something a chatbot cannot.

The guidance below is ordinary food-safety advice, not a standard. It is worded
as ranges for that reason, and the reply says where it comes from.
"""
from __future__ import annotations

import re

# word in the sentence -> label, fresh compartment, freezer
FOOD_TABLE: list[dict] = [
    {"keys": ["chicken", "hen", "poultry"],
     "label": "raw chicken",
     "fresh": "one to two days",
     "frozen": "several months"},
    {"keys": ["beef", "pork", "lamb", "meat", "mince", "minced beef",
              "ground beef", "sausage", "bacon"],
     "label": "raw meat",
     "fresh": "one to two days",
     "frozen": "several months"},
    {"keys": ["fish", "salmon", "tuna", "cod", "prawn", "prawns", "shrimp",
              "seafood", "crab"],
     "label": "raw fish",
     "fresh": "about a day",
     "frozen": "several months"},
    {"keys": ["milk", "cream"],
     "label": "milk",
     "fresh": "a few days, and the printed date decides",
     "frozen": "it can be frozen but separates when thawed"},
    {"keys": ["egg", "eggs"],
     "label": "eggs",
     "fresh": "three to five weeks",
     "frozen": "not in the shell"},
    {"keys": ["cooked rice", "leftover rice", "boiled rice", "rice"],
     "label": "cooked rice",
     "fresh": "about a day, so it is the one to be careful with",
     "frozen": "a month or more"},
    {"keys": ["leftovers", "leftover", "takeaway", "takeout"],
     "label": "leftovers",
     "fresh": "one to two days",
     "frozen": "a month or more"},
    {"keys": ["butter"],
     "label": "butter",
     "fresh": "a few weeks once opened",
     "frozen": "it freezes well"},
    {"keys": ["cheese"],
     "label": "cheese",
     "fresh": "a few weeks once opened, and a hard cheese keeps longer",
     "frozen": "it can be frozen, but it crumbles when thawed"},
    {"keys": ["vegetable", "vegetables", "veggies", "lettuce", "spinach",
              "carrot", "carrots", "broccoli", "cabbage", "cucumber", "salad",
              "greens", "herb", "herbs", "mushroom", "mushrooms", "onion",
              "onions", "potato", "potatoes"],
     "label": "most vegetables",
     "fresh": "three to seven days, and they keep better away from the coldest "
              "part of the compartment",
     "frozen": "they need blanching first"},
    {"keys": ["fruit", "fruits", "berries", "strawberry", "strawberries",
              "banana", "bananas", "apple", "apples", "orange", "oranges",
              "grape", "grapes"],
     "label": "most fruit",
     "fresh": "three to seven days",
     "frozen": "they freeze well"},
    {"keys": ["ice cream", "icecream"],
     "label": "ice cream",
     "fresh": "it does not belong there",
     "frozen": "that is what it is for"},
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
        return False
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
    subject = label if qualifier in ("raw", "cooked") else matched
    return subject[:1].upper() + subject[1:]


def _verb(subject: str) -> str:
    """keeps / keep — "Raw chicken keeps", "Vegetables keep"."""
    return "keep" if subject.lower().endswith("s") else "keeps"


def _note() -> str:
    """Where the guidance comes from. Short enough to survive being repeated."""
    return "(general guidance, not a reading from the fridge)"


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
    food = match_food(text)
    if food is None:
        return ""

    name, fresh_temp, frozen_temp = "", None, None
    if fridge:
        name = fridge.get("name", "the refrigerator")
        fresh_temp = (fridge.get("state") or {}).get("fridge_temp")
        frozen_temp = (fridge.get("state") or {}).get("freezer_temp")

    lines: list[str] = []
    if name and fresh_temp is not None:
        if frozen_temp is not None:
            lines.append(f"{name}: {fresh_temp} °C · freezer {frozen_temp} °C")
        else:
            lines.append(f"{name}: {fresh_temp} °C")

    subject = _subject(food)

    if wants_adequacy(text) and not wants_storage(text):
        # A judgement, not a comparison. If the user named a temperature, judge
        # that one and say what the house is actually at; otherwise judge the
        # reading. Answering the measured number at a question that asked about
        # 4 °C is the same failure as answering a different question entirely.
        m = DEGREES_RE.search(text or "")
        asked = int(m.group(1)) if m else None
        judge, source = fresh_temp, f"{fresh_temp} °C"
        if asked is not None and 0 <= asked <= 30:
            judge, source = asked, f"{asked} °C"
        elif fresh_temp is None:
            lines.append("This house has no refrigerator I can read, so I can "
                         "only judge the guidance, not your appliance.")
            lines.append(f"{subject}: {food['fresh']} in a fresh-food compartment.")
            return "\n".join(lines)
        verdict = ("cold enough for that" if judge is not None and judge <= 4
                   else "warmer than that likes")
        lines.append(f"{subject} {_verb(subject)} {food['fresh']}, "
                     f"so {source} is {verdict}.")
        if asked is not None and fresh_temp is not None and asked != fresh_temp:
            lines.append(f"Right now the compartment is at {fresh_temp} °C.")
        if note:
            lines.append(_note())
        return "\n".join(lines)

    lines.append(f"{subject} — fridge: {food['fresh']} · freezer: {food['frozen']}")
    if not name:
        lines.append("I cannot read a refrigerator here, so this is the general "
                     "rule rather than your appliance.")
    elif note:
        lines.append(_note())
    return "\n".join(lines)
