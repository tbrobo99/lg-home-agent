"""Named routines for requests that are about the home, not about a device.

"Wake the house up" is not a device command. It is a routine: several devices,
different actions, and one rule nobody states out loud — the fridge stays on.

Neither existing tier can do this well. The rule engine cannot parse it, because
there is no device called "the house". A model given five devices and a flat
action list answers by asking which device is meant, or by switching the fridge
off. The benchmark caught both, and both were reported as model failures when
they were really a missing layer.

So it is neither: a scene is a preset. Detection is a phrase match, expansion is
a table, and both are deterministic — microseconds, no network, and no way to
invent a device. This is the same trade the rule engine already makes for times;
it just applies to routines instead of clock arithmetic.

The table is per device *type*, not per device id, so adding a second bedroom AC
needs no change here.
"""
from __future__ import annotations

import re

# Types a scene never switches off. "Everything off" is a figure of speech: it
# has never meant the fridge, and a fridge that is off is spoiled food.
KEEP_RUNNING = {"refrigerator"}

SCENES: dict[str, dict] = {
    "morning": {
        "label": "Good morning",
        "by_type": {
            "air_conditioner": [{"action": "power", "value": True}],
            "air_purifier": [{"action": "power", "value": True}],
            "dehumidifier": [{"action": "power", "value": True}],
        },
        # The washer and the dryer are not part of waking up.
        "default": None,
    },
    "night": {
        "label": "Good night",
        "by_type": {},
        "default": [{"action": "power", "value": False}],
    },
    "away": {
        "label": "Away",
        "by_type": {},
        "default": [{"action": "power", "value": False}],
    },
}

# Ordered: the first match wins, and "good morning" is more specific than
# "wake ... up".
TRIGGERS: tuple[tuple[str, re.Pattern], ...] = (
    ("morning", re.compile(r"\bgood\s+morning\b|\bwake\b[^.?!]{0,24}?\bup\b", re.I)),
    ("night", re.compile(
        r"\bgood\s*night\b|\bbed\s*time\b"
        r"|\b(?:put|get|set|tuck)\b[^.?!]{0,28}?"
        r"\b(?:to\s+sleep|to\s+bed|ready\s+for\s+bed)\b", re.I)),
    ("away", re.compile(
        r"\b(?:i'?m|i\s+am)\s+(?:leaving|going\s+out|heading\s+out|off\s+out)\b"
        r"|\bleaving\s+now\b|\bgoing\s+out\b|\bheading\s+out\b"
        r"|\baway\s+mode\b", re.I)),
)


def detect(text: str) -> str | None:
    """Return the scene a sentence names, or None.

    A phrase match, deliberately: a model asked to classify "wake the house up"
    gets it right most of the time, and a routine that works most of the time is
    not a routine.
    """
    t = text or ""
    for key, rx in TRIGGERS:
        if rx.search(t):
            return key
    return None


def label(key: str) -> str:
    return SCENES.get(key, {}).get("label", key)


def actions_for(key: str, device: dict) -> list[dict] | None:
    """The actions a scene applies to one device, or None to leave it alone."""
    spec = SCENES[key]
    dtype = device.get("type") or ""

    if dtype in spec["by_type"]:
        return [dict(a) for a in spec["by_type"][dtype]]
    if dtype in KEEP_RUNNING:
        return None
    default = spec["default"]
    return [dict(a) for a in default] if default else None


def expand(key: str, devices: list[dict]) -> list[tuple[dict, list[dict]]]:
    """Pair each device the scene touches with the actions it gets."""
    out = []
    for d in devices:
        acts = actions_for(key, d)
        if acts:
            out.append((d, acts))
    return out
