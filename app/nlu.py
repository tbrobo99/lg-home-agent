"""Natural-language understanding (rule engine).

Why rules before an LLM:

1. Zero dependencies, zero cost, works offline — the demo runs on any laptop.
2. Time parsing has to be exactly right. Misreading "half past ten" turns the
   AC on at the wrong hour, and that is not a failure mode you want a
   probabilistic model to own.
3. It is deterministic and testable, so the LLM only has to handle the long
   tail ("it feels stuffy in here") instead of the common cases.

Everything returns the same intent dict, which agent.py executes.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

# ---------------------------------------------------------------- numbers

WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20,
}
NUMW = r"(?:\d{1,2}|" + "|".join(WORD_NUM) + r")"


def to_num(s: str) -> int | None:
    s = (s or "").strip().lower()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s in ("a", "an"):
        return 1
    return WORD_NUM.get(s)


# ---------------------------------------------------------------- time

COLON_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?", re.I)
AT_HOUR_RE = re.compile(
    r"\bat\s+(" + NUMW + r")(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.|o'?clock)?", re.I
)
MERIDIEM_RE = re.compile(r"\b(" + NUMW + r")(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", re.I)
HALF_PAST_RE = re.compile(r"\bhalf\s+past\s+(" + NUMW + r")", re.I)
QUARTER_PAST_RE = re.compile(r"\b(?:a\s+)?quarter\s+past\s+(" + NUMW + r")", re.I)
QUARTER_TO_RE = re.compile(r"\b(?:a\s+)?quarter\s+to\s+(" + NUMW + r")", re.I)
IN_HALF_RE = re.compile(r"\bin\s+(?:half\s+an?|a\s+half)\s*(?:hour|hr)\b", re.I)
IN_N_RE = re.compile(
    r"\bin\s+(" + NUMW + r"|an?)\s*(hours?|hrs?|minutes?|mins?)", re.I
)

DAY_WORDS = [
    ("monday", 0), ("mon", 0), ("tuesday", 1), ("tues", 1), ("tue", 1),
    ("wednesday", 2), ("weds", 2), ("wed", 2), ("thursday", 3), ("thurs", 3),
    ("thur", 3), ("thu", 3), ("friday", 4), ("fri", 4), ("saturday", 5),
    ("sat", 5), ("sunday", 6), ("sun", 6),
]
DAY_MAP = dict(DAY_WORDS)
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_RE = re.compile(r"\b(" + "|".join(w for w, _ in DAY_WORDS) + r")\b", re.I)


def find_days(text: str) -> list[int]:
    days: list[int] = []
    for m in DAY_RE.finditer(text):
        d = DAY_MAP.get(m.group(1).lower())
        if d is not None and d not in days:
            days.append(d)
    return sorted(days)


def _period_word(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(morning|sunrise)\b", t):
        return "am"
    if re.search(r"\b(afternoon|evening|tonight|tonite|night)\b", t):
        return "pm"
    if re.search(r"\bnoon\b", t):
        return "noon"
    if re.search(r"\bmidnight\b", t):
        return "midnight"
    return None


def _apply_meridiem(hour: int, meridiem: str | None, now: datetime) -> int:
    if meridiem == "noon":
        return 12
    if meridiem == "midnight":
        return 0
    if meridiem in ("pm", "p.m."):
        return (hour % 12) + 12 if hour < 12 else hour
    if meridiem in ("am", "a.m."):
        return 0 if hour == 12 else hour
    if hour == 12:
        return 12
    # No explicit am/pm: if it is already afternoon, assume the coming pm hour.
    # "at 10" said at 18:42 means 22:00, not 10:00 tomorrow.
    if now.hour >= 12 and hour <= 11:
        return hour + 12
    return hour


def extract_time(text: str, now: datetime) -> dict | None:
    """Return {'kind':'relative','seconds':n} or {'kind':'clock','hour','minute'}."""
    if IN_HALF_RE.search(text):
        return {"kind": "relative", "seconds": 1800}

    m = IN_N_RE.search(text)
    if m:
        n = to_num(m.group(1))
        if n:
            unit = m.group(2).lower()
            return {"kind": "relative",
                    "seconds": n * (3600 if unit.startswith(("hour", "hr")) else 60)}

    m = HALF_PAST_RE.search(text)
    if m:
        h = to_num(m.group(1))
        if h is not None:
            return {"kind": "clock", "hour": h, "minute": 30, "explicit": None}

    m = QUARTER_PAST_RE.search(text)
    if m:
        h = to_num(m.group(1))
        if h is not None:
            return {"kind": "clock", "hour": h, "minute": 15, "explicit": None}

    m = QUARTER_TO_RE.search(text)
    if m:
        h = to_num(m.group(1))
        if h is not None:
            return {"kind": "clock", "hour": (h - 1) % 24, "minute": 45, "explicit": None}

    for rx in (COLON_RE, AT_HOUR_RE, MERIDIEM_RE):
        m = rx.search(text)
        if m:
            h = int(m.group(1)) if m.group(1).isdigit() else to_num(m.group(1))
            if h is None:
                continue
            minute = int(m.group(2)) if m.group(2) else 0
            return {"kind": "clock", "hour": h % 24, "minute": minute,
                    "explicit": m.group(3)}
    return None


def detect_repeat(text: str) -> dict | None:
    """Detect recurrence."""
    m = re.search(r"\bevery\s+(" + NUMW + r"|few)\s*(hours?|hrs?|minutes?|mins?)",
                  text, re.I)
    if m:
        n = to_num(m.group(1)) or 1
        unit = m.group(2).lower()
        return {"type": "interval",
                "interval_seconds": n * (3600 if unit.startswith(("hour", "hr")) else 60)}

    if re.search(r"\bevery\s+hour\b|\bhourly\b", text, re.I):
        return {"type": "interval", "interval_seconds": 3600}

    if re.search(r"\b(every\s+day|daily|each\s+day|every\s+night)\b", text, re.I):
        return {"type": "daily"}

    if re.search(r"\b(every\s+)?weekdays?\b|\bworkdays?\b", text, re.I):
        return {"type": "weekday"}

    if re.search(r"\bevery\s+weekend\b|\bon\s+weekends?\b", text, re.I):
        return {"type": "weekly", "weekdays": [5, 6]}

    days = find_days(text)
    if days and re.search(r"\b(every|each)\b", text, re.I):
        return {"type": "weekly", "weekdays": days}

    return None


def compute_next(rule: dict, now: datetime) -> datetime | None:
    """Next execution time for a rule, or None if the rule is unusable."""
    t = rule.get("type")
    if t == "interval":
        return now + timedelta(seconds=int(rule["interval_seconds"]))
    if t == "once":
        try:
            return datetime.fromisoformat(rule["run_at"])
        except Exception:
            return None

    at = rule.get("at") or "00:00"
    hh, mm = (int(x) for x in at.split(":"))
    cand = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if cand <= now:
        cand += timedelta(days=1)

    if t == "daily":
        return cand
    if t == "weekday":
        days = [0, 1, 2, 3, 4]
    elif t == "weekly":
        days = rule.get("weekdays") or list(range(7))
    else:
        return cand

    for i in range(0, 8):
        d = cand + timedelta(days=i)
        if d.weekday() in days:
            return d
    return cand


def _fmt_once(dt: datetime) -> str:
    return dt.strftime("%b %d, %H:%M")


def build_rule(text: str, now: datetime) -> tuple[dict | None, str]:
    """Combine recurrence and clock time into (rule, human-readable label)."""
    repeat = detect_repeat(text)
    tm = extract_time(text, now)

    if repeat and repeat["type"] == "interval":
        sec = repeat["interval_seconds"]
        label = (f"every {sec // 3600} h" if sec >= 3600
                 else f"every {sec // 60} min")
        return {"type": "interval", "interval_seconds": sec}, label

    if tm and tm["kind"] == "relative":
        run_at = now + timedelta(seconds=tm["seconds"])
        label = (f"in {tm['seconds'] // 60} min" if tm["seconds"] < 3600
                 else f"in {tm['seconds'] // 3600} h")
        return {"type": "once", "run_at": run_at.isoformat(timespec="seconds")}, label

    if not tm:
        return None, ""

    meridiem = tm.get("explicit") or _period_word(text)
    hour = _apply_meridiem(tm["hour"], meridiem, now)
    minute = tm["minute"]
    at = f"{hour:02d}:{minute:02d}"

    if not repeat:
        days = find_days(text)
        base = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if re.search(r"\btomorrow\b", text, re.I):
            base += timedelta(days=1)
        elif days:
            # "Monday at 7" means the next Monday, not today
            target = days[0]
            while base.weekday() != target:
                base += timedelta(days=1)
        if base <= now:
            base += timedelta(days=1)
            if days:
                target = days[0]
                while base.weekday() != target:
                    base += timedelta(days=1)
        return {"type": "once", "run_at": base.isoformat(timespec="seconds")}, _fmt_once(base)

    rule: dict = {"type": repeat["type"], "at": at}
    if repeat["type"] == "weekly":
        rule["weekdays"] = repeat["weekdays"]
        label = f"every {', '.join(DAY_NAMES[d] for d in repeat['weekdays'])} at {at}"
    elif repeat["type"] == "weekday":
        label = f"every weekday at {at}"
    else:
        label = f"every day at {at}"
    return rule, label


# ---------------------------------------------------------------- devices

TYPE_ALIASES = {
    "air_conditioner": [r"\bac\b", r"\ba/c\b", r"air[\s-]?con(?:ditioner|ditioning)?",
                        r"\baircon\b"],
    "air_purifier": [r"\bpurifier\b", r"air purifier"],
    "washer": [r"\bwasher\b", r"washing machine"],
    "dryer": [r"\bdryer\b", r"tumble dryer"],
    "refrigerator": [r"\bfridge\b", r"\brefrigerator\b"],
    "dish_washer": [r"\bdishwasher\b"],
    "water_heater": [r"water heater"],
    "dehumidifier": [r"\bdehumidifier\b"],
    "styler": [r"\bstyler\b", r"clothes care"],
    "robot_cleaner": [r"robot vacuum", r"\broombra\b", r"vacuum"],
}

_ALIAS_CACHE: dict[str, list[re.Pattern]] = {}


def _alias_patterns(dtype: str) -> list[re.Pattern]:
    if dtype not in _ALIAS_CACHE:
        _ALIAS_CACHE[dtype] = [re.compile(p, re.I)
                               for p in TYPE_ALIASES.get(dtype, [])]
    return _ALIAS_CACHE[dtype]


def match_device(text: str, devices: list[dict], default_room: str = "",
                 fallback: dict | None = None) -> tuple[dict | None, list[dict]]:
    """Return (best match, tied candidates)."""
    text_l = text.lower()
    scored: list[tuple[int, dict]] = []

    for d in devices:
        score = 0
        name = (d.get("name") or "").lower()
        room = (d.get("room") or "").lower()
        if name and name in text_l:
            score = 100
        else:
            type_hit = any(p.search(text_l) for p in _alias_patterns(d.get("type", "")))
            room_hit = bool(room) and room in text_l
            if type_hit and room_hit:
                score = 80
            elif type_hit:
                score = 50
            elif room_hit:
                score = 25
        if score and room and room not in text_l:
            score -= 3
        if score > 0:
            scored.append((score, d))

    if not scored:
        if fallback:
            return fallback, []
        return None, []

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[0][0]
    ties = [d for s, d in scored if s == top]

    if len(ties) > 1 and default_room:
        for d in ties:
            if (d.get("room") or "").lower() == default_room.lower():
                return d, ties
    if len(ties) > 1:
        online = [d for d in ties if d.get("online", True)]
        if len(online) == 1:
            return online[0], ties
        return ties[0], ties
    return ties[0], []


# ---------------------------------------------------------------- actions

OFF_RE = re.compile(
    r"\b(turn|switch|power)\s+(?:it\s+|them\s+)?off\b"
    r"|\bshut\s+(?:it\s+|them\s+)?(?:down|off)\b"
    r"|\b(stop|kill|deactivate|disable)\b"
    r"|\bswitch\s+off\b", re.I)
ON_RE = re.compile(
    r"\b(turn|switch|power)\s+(?:it\s+|them\s+)?on\b"
    r"|\b(start|activate|enable|resume)\b"
    r"|\bwake\s+up\b|\bswitch\s+on\b", re.I)

TEMP_DEG_RE = re.compile(
    r"\b(\d{1,2})\s*(?:\u00b0\s*c|\u00b0|degrees?(?:\s+celsius)?|celsius)\b", re.I)
TEMP_SET_RE = re.compile(r"\b(?:set|change|adjust|put|turn)\b[^.?!]{0,24}?\bto\s+(\d{1,2})\b", re.I)
TEMP_MAKE_RE = re.compile(r"\bmake\s+it\s+(\d{1,2})\b", re.I)

MODE_PATTERNS = [
    (re.compile(r"\b(cool|cooling|air\s*condition(?:ing)?|chill)\b", re.I), "cool"),
    (re.compile(r"\b(heat|heating|warm\s+up)\b", re.I), "heat"),
    (re.compile(r"\b(dry|dehumidif\w*)\b", re.I), "dry"),
    (re.compile(r"\bfan\s*(?:only|mode)\b", re.I), "fan"),
    (re.compile(r"\bauto\s*mode\b", re.I), "auto"),
]
FAN_PATTERNS = [
    (re.compile(r"\bfan\s*(?:speed\s*)?(?:high|max|strong)\b|\bhigh\s+fan\b", re.I), "high"),
    (re.compile(r"\bfan\s*(?:speed\s*)?(?:mid|medium|normal)\b", re.I), "mid"),
    (re.compile(r"\bfan\s*(?:speed\s*)?(?:low|min|quiet)\b|\blow\s+fan\b", re.I), "low"),
    (re.compile(r"\bfan\s*(?:speed\s*)?auto\b|\bauto\s+fan\b", re.I), "auto"),
]


def extract_actions(text: str) -> list[dict]:
    acts: list[dict] = []

    off = bool(OFF_RE.search(text))
    on = bool(ON_RE.search(text))
    if off:
        acts.append({"action": "power", "value": False})
    elif on:
        acts.append({"action": "power", "value": True})

    temp = None
    m = TEMP_DEG_RE.search(text)
    if m:
        temp = int(m.group(1))
    if temp is None:
        m = TEMP_SET_RE.search(text) or TEMP_MAKE_RE.search(text)
        if m:
            temp = int(m.group(1))
    if temp is not None and 16 <= temp <= 30:
        acts.append({"action": "temperature", "value": temp})

    for rx, mode in MODE_PATTERNS:
        if rx.search(text):
            acts.append({"action": "mode", "value": mode})
            break

    for rx, fan in FAN_PATTERNS:
        if rx.search(text):
            acts.append({"action": "fan", "value": fan})
            break

    return acts


# ---------------------------------------------------------------- entry point

QUERY_RE = re.compile(
    r"\bis\s+(?:it|the\s+[\w\s]{0,18}?)\s+(?:on|off|running|working)\b"
    r"|\bwhat(?:'s|\s+is)\s+the\s+(?:temperature|temp|status|state)\b"
    r"|\bhow\s+(?:hot|cold|warm)\b"
    r"|\bwhat(?:'s|\s+is)\s+(?:it|the\s+[\w\s]{0,18}?)\s+set\s+to\b"
    r"|\bcurrent\s+(?:temperature|temp|status|state)\b"
    r"|\b(?:status|state)\s+(?:of|report)\b"
    r"|\bis\s+everything\s+(?:off|on)\b"
    r"|\bwhat(?:'s|\s+is)\s+(?:it|the\s+\w+)\s+doing\b", re.I)

CANCEL_RE = re.compile(
    r"\b(cancel|delete|remove|clear|drop)\b.*\b(schedule[sd]?|timer[s]?|task[s]?|job[s]?|reminder[s]?)\b"
    r"|\b(schedule[sd]?|timer[s]?|task[s]?|job[s]?)\b.*\b(cancel|delete|remove|clear)\b", re.I)

LIST_RE = re.compile(
    r"\b(what|which|list|show|any)\b.*\b(schedule[sd]?|timer[s]?|task[s]?|job[s]?|reminder[s]?)\b"
    r"|\b(schedule[sd]?|timer[s]?|task[s]?|job[s]?)\b.*\b(list|do i have|are there)\b", re.I)


def parse(text: str, devices: list[dict], now: datetime | None = None,
          default_room: str = "Living Room",
          fallback_device: dict | None = None) -> dict:
    now = now or datetime.now()
    t = (text or "").strip()
    res = {
        "intent": "unknown", "device_id": None, "device_name": None,
        "actions": [], "rule": None, "repeat_label": "",
        "confidence": 0.0, "clarify": None, "raw_text": t,
    }
    if not t:
        res["clarify"] = 'Say something like "turn off the living room AC at 11 pm".'
        return res

    # 1. cancel / list scheduled tasks
    if CANCEL_RE.search(t):
        res["intent"] = "cancel"
        res["confidence"] = 0.9
        res["cancel_all"] = bool(re.search(r"\ball\b|\beverything\b|\bclear\b", t, re.I))
        return res

    if LIST_RE.search(t):
        res["intent"] = "list"
        res["confidence"] = 0.9
        return res

    # 2. actions + time decide control vs schedule
    actions = extract_actions(t)
    rule, label = build_rule(t, now)
    device, ties = match_device(t, devices, default_room, fallback_device)

    if rule:
        if not actions:
            res["clarify"] = ('What should I do at that time? '
                              'e.g. "turn off the AC at 11 pm".')
            return res
        if not device:
            res["intent"] = "unknown"
            res["clarify"] = ("Which device? I can control: "
                              + ", ".join(d["name"] for d in devices[:6]))
            return res
        res.update(intent="schedule", device_id=device["id"],
                   device_name=device["name"], actions=actions,
                   rule=rule, repeat_label=label, confidence=0.92)
        if len(ties) > 1:
            res["clarify"] = (f'Multiple devices matched, so I used "{device["name"]}".')
        return res

    if actions:
        if not device:
            res["intent"] = "unknown"
            res["clarify"] = ("Which device? I can control: "
                              + ", ".join(d["name"] for d in devices[:6]))
            return res
        res.update(intent="control", device_id=device["id"],
                   device_name=device["name"], actions=actions, confidence=0.88)
        if len(ties) > 1:
            res["clarify"] = (f'Multiple devices matched, so I used "{device["name"]}".')
        return res

    # 3. status questions
    if QUERY_RE.search(t):
        device, ties = match_device(t, devices, default_room, fallback_device)
        res.update(intent="query", confidence=0.85)
        if not device:
            pool = ([d for d in devices
                     if (d.get("room") or "").lower() == default_room.lower()]
                    or [d for d in devices if d.get("type") == "air_conditioner"]
                    or devices)
            device = pool[0] if pool else None
            ties = []
        if device:
            res["device_id"] = device["id"]
            res["device_name"] = device["name"]
            if len(ties) > 1:
                res["clarify"] = (f'Multiple devices matched, so I answered for '
                                  f'"{device["name"]}".')
        else:
            res["clarify"] = "I could not tell which device you mean."
        return res

    # 4. fallback
    res["clarify"] = ('I can turn devices on and off, set the temperature, and '
                      'schedule things. Try "set the living room AC to 26 degrees" '
                      'or "turn off the living room AC every night at 11 pm".')
    return res
