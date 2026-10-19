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
    {"keys": ["butter", "cheese"],
     "label": "butter and cheese",
     "fresh": "a few weeks once opened",
     "frozen": "cheese can be frozen for cooking"},
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
    """Return the first entry whose trigger word appears in the sentence."""
    t = (text or "").lower()
    for entry in FOOD_TABLE:
        for key in entry["keys"]:
            if re.search(r"\b" + re.escape(key) + r"\b", t):
                return entry
    return None


def wants_storage(text: str) -> bool:
    return bool(STORAGE_INTENT_RE.search(text or ""))


def wants_adequacy(text: str) -> bool:
    return bool(ADEQUACY_RE.search(text or ""))


def is_food_request(text: str) -> bool:
    """True when the sentence is about storing food rather than about a device.

    Guarded on purpose. "turn the fridge down to 3" mentions no food, so it
    stays a control command; "can I keep the chicken in the fridge" names food
    and a place, so it never reaches the comfort rules, which would read the
    word "cold" as a request to heat the room.
    """
    if not text:
        return False
    return match_food(text) is not None and (
        wants_storage(text) or wants_adequacy(text))


# "what temperature is that" asked with a number in it, e.g. "is 4 degrees
# enough". The user is asking about a value, so answer about that value.
DEGREES_RE = re.compile(r"\b(\d{1,2})\s*(?:°\s*c|°|degrees?(?:\s+celsius)?|"
                        r"celsius)\b", re.I)


def _quote(food: dict) -> str:
    """Both compartments as two labelled phases.

    Labelled rather than written as one sentence, because the frozen entries are
    not all durations: "several months" is a span, "not in the shell" is a note,
    and a single verb cannot carry both without producing "it keeps not in the
    shell".
    """
    return (f'fresh compartment: {food["fresh"]}. freezer: {food["frozen"]}.')


def _verb(food: dict) -> str:
    """keeps / keep — "eggs keep", "raw chicken keeps"."""
    return "keep" if food["label"].endswith("s") else "keeps"


def answer(text: str, fridge: dict | None) -> str:
    """Compose the reply: the measured temperature first, then the guidance.

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

    parts: list[str] = []
    if name and fresh_temp is not None:
        if frozen_temp is not None:
            parts.append(f"{name} is running at {fresh_temp} °C with a "
                         f"freezer at {frozen_temp} °C.")
        else:
            parts.append(f"{name} is running at {fresh_temp} °C.")

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
            parts.append("This house has no refrigerator I can read, so I can "
                         "only judge the guidance, not your appliance.")
            parts.append(f"{food['label'].capitalize()} {_verb(food)} "
                         f"{food['fresh']} in a fresh-food compartment.")
            return " ".join(parts)
        verdict = ("cold enough for that" if judge is not None and judge <= 4
                   else "warmer than that likes")
        parts.append(f"{food['label'].capitalize()} {_verb(food)} "
                     f"{food['fresh']}, so {source} is {verdict}.")
        if asked is not None and fresh_temp is not None and asked != fresh_temp:
            parts.append(f"Right now the compartment is at {fresh_temp} °C.")
        parts.append("This is general food-safety guidance, not a figure the "
                     "appliance reports.")
        return " ".join(parts)

    parts.append(f"{food['label'].capitalize()}: {_quote(food)}")
    if not name:
        parts.append("I cannot read a refrigerator here, so this is the "
                     "general rule rather than your appliance.")
    else:
        parts.append("This is general food-safety guidance, not a figure the "
                     "appliance reports.")
    return " ".join(parts)
