"""Tools the model may call when the rule engine gives up.

Three rules hold this layer together, and all three exist because of a failure
mode that actually happened while building it:

1. **The model never supplies a timestamp.** `schedule` takes the user's own
   words and hands them to `nlu.build_rule`, which is deterministic and covered
   by the benchmark. A model that invents "23:30" cannot reach the job table.
2. **An invalid device id is a recoverable error, not an exception.** Asked to
   turn off the bedroom AC, `glm-4-flash` answered `bedroom_ac` when the real id
   is `ac_bedroom`. The tool replies with the valid ids so the next step can fix
   itself — which is what makes it a loop instead of a pipeline.
3. **Every call is validated here, not in the adapter.** The adapter's job is
   talking to appliances; the toolbox's job is refusing nonsense before it gets
   that far.
"""
from __future__ import annotations

import re
from datetime import datetime

from . import db, nlu
from .devices.base import (FAN_LABELS, MODE_LABELS, describe_action,
                           normalize_action)

ALLOWED_ACTIONS = ("power", "temperature", "mode", "fan")
_TRUE = {"true", "on", "yes", "1"}
_FALSE = {"false", "off", "no", "0"}

# Small models sometimes echo the tool name into its own argument
# ("finish\nThe AC is on"), and sometimes into prose instead of calling the
# tool at all. Cheap to clean, ugly to show a user.
_ECHOED_NAME = re.compile(r"^(?:finish|ask_user)\b[\s:：\-]*", re.I)


def clean_reply(text: str) -> str:
    """Strip a tool name the model pasted in front of its own sentence."""
    return _ECHOED_NAME.sub("", (text or "").strip()).strip()

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_devices",
            "description": "List every controllable device with its id, room and "
                           "whether it is currently on. Call this first if you are "
                           "not sure which device the user means.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": "Read the current state of one device: power, target "
                           "temperature, mode, fan speed, room temperature.",
            "parameters": {
                "type": "object",
                "properties": {"device_id": {"type": "string"}},
                "required": ["device_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control",
            "description": "Change a device right now. Use one call per device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": list(ALLOWED_ACTIONS)},
                                "value": {
                                    "description": "true/false for power, an integer "
                                                   "16-30 for temperature, a string for "
                                                   "mode or fan",
                                },
                            },
                            "required": ["action", "value"],
                        },
                    },
                },
                "required": ["device_id", "actions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule",
            "description": "Create a task that runs later. Pass the user's own words "
                           "for the time — do not convert it to a clock time yourself, "
                           "a deterministic parser does that.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": list(ALLOWED_ACTIONS)},
                                "value": {},
                            },
                            "required": ["action", "value"],
                        },
                    },
                    "when_text": {
                        "type": "string",
                        "description": "The time as the user said it, e.g. "
                                       "'every night at 11 pm' or 'tomorrow at 7'",
                    },
                },
                "required": ["device_id", "actions", "when_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Stop and ask one short question when the request is "
                           "ambiguous or missing something you cannot look up.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "End the turn with a short sentence for the user. Call this "
                           "after the tools have done their work.",
            "parameters": {
                "type": "object",
                "properties": {"reply": {"type": "string"}},
                "required": ["reply"],
            },
        },
    },
]


class ToolBox:
    """Executes tool calls and accumulates what the turn actually did."""

    def __init__(self, adapter, cfg: dict, now: datetime, default_room: str = "") -> None:
        self.adapter = adapter
        self.cfg = cfg
        self.now = now
        self.default_room = default_room

        self.actions: list[dict] = []
        self.targets: list[dict] = []
        self.reply: str | None = None
        self.question: str | None = None
        self.finished = False
        self.calls = 0

    # ------------------------------------------------ dispatch

    def call(self, name: str, args: dict) -> dict:
        self.calls += 1
        handler = getattr(self, "_t_" + str(name), None)
        if handler is None:
            return {"error": f"no such tool: {name}",
                    "available": ["list_devices", "get_state", "control",
                                  "schedule", "ask_user", "finish"]}
        if not isinstance(args, dict):
            return {"error": "arguments must be an object"}
        try:
            return handler(**args)
        except TypeError as exc:
            return {"error": f"bad arguments: {exc}"}
        except Exception as exc:  # a tool must never take the turn down with it
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _devices(self) -> list[dict]:
        return self.adapter.list_devices()

    def _state(self, device_id: str) -> dict:
        try:
            return self.adapter.get_state(device_id) or {}
        except Exception:
            return {}

    def _find(self, device_id: str) -> dict | None:
        for d in self._devices():
            if d["id"] == device_id:
                return d
        return None

    def _unknown_device(self, device_id: str) -> dict:
        known = [{"id": d["id"], "name": d["name"]} for d in self._devices()]
        return {"error": f"no device with id {device_id!r}", "devices": known}

    def _clean_actions(self, actions) -> tuple[list[dict], str | None]:
        """Coerce what the model sends into the shapes the adapter accepts.

        Small models routinely emit `"power": "false"` or `"temperature": "24"`.
        Rejecting those would burn a step and teach nothing, so they are coerced;
        only values that cannot be read unambiguously are refused.
        """
        if not isinstance(actions, list) or not actions:
            return [], "actions must be a non-empty array"
        out = []
        for raw in actions:
            if not isinstance(raw, dict):
                return [], "each action must be an object"
            act = normalize_action(raw)
            name, value = act["action"], act.get("value")

            if name not in ALLOWED_ACTIONS:
                return [], (f"unknown action {name!r}; use one of "
                            f"{', '.join(ALLOWED_ACTIONS)}")

            if name == "power":
                if isinstance(value, bool):
                    pass
                elif isinstance(value, str) and value.strip().lower() in _TRUE:
                    value = True
                elif isinstance(value, str) and value.strip().lower() in _FALSE:
                    value = False
                elif isinstance(value, (int, float)) and value in (0, 1):
                    value = bool(value)
                else:
                    return [], "power must be true or false"

            elif name == "temperature":
                try:
                    value = int(round(float(value)))
                except (TypeError, ValueError):
                    return [], "temperature must be a whole number"
                if not 16 <= value <= 30:
                    return [], "temperature must be between 16 and 30"

            elif name == "mode":
                value = str(value).strip().lower()
                if value not in MODE_LABELS:
                    return [], f"mode must be one of {', '.join(MODE_LABELS)}"

            elif name == "fan":
                value = str(value).strip().lower()
                if value not in FAN_LABELS:
                    return [], f"fan must be one of {', '.join(FAN_LABELS)}"

            out.append({"action": name, "value": value})
        return out, None

    # ------------------------------------------------ tools

    def _t_list_devices(self) -> dict:
        out = []
        for d in self._devices():
            state = self._state(d["id"])
            out.append({
                "id": d["id"],
                "name": d["name"],
                "room": d.get("room", ""),
                "type": d.get("type_label") or d.get("type"),
                "online": d.get("online", True),
                "power": state.get("power"),
            })
        return {"devices": out}

    def _t_get_state(self, device_id: str = "") -> dict:
        d = self._find(device_id)
        if not d:
            return self._unknown_device(device_id)
        return {
            "device": d["name"],
            "room": d.get("room", ""),
            "online": d.get("online", True),
            "capabilities": d.get("capabilities") or [],
            "state": self._state(d["id"]),
        }

    def _t_control(self, device_id: str = "", actions=None) -> dict:
        d = self._find(device_id)
        if not d:
            return self._unknown_device(device_id)
        if not d.get("online", True):
            return {"error": f"{d['name']} is offline, nothing was sent"}

        clean, err = self._clean_actions(actions)
        if err:
            return {"error": err}

        res = self.adapter.apply(d["id"], clean)
        if not res.get("ok"):
            return {"error": res.get("error") or "the device rejected the command",
                    "device": d["name"]}

        self.actions.extend({**a, "device_id": d["id"], "device_name": d["name"]}
                            for a in clean)
        self.targets.append({"id": d["id"], "name": d["name"], "actions": clean})
        db.log("tool.control", f"{d['name']} {clean} -> ok")
        return {
            "ok": True,
            "device": d["name"],
            "did": ", ".join(describe_action(a) for a in clean),
            "state": self._state(d["id"]),
        }

    def _t_schedule(self, device_id: str = "", actions=None, when_text: str = "") -> dict:
        d = self._find(device_id)
        if not d:
            return self._unknown_device(device_id)

        clean, err = self._clean_actions(actions)
        if err:
            return {"error": err}

        # The guardrail: the model hands over the words, not a timestamp.
        rule, label = nlu.build_rule(str(when_text or ""), self.now)
        if not rule:
            return {
                "error": f"could not read a time from {when_text!r}",
                "hint": "pass the user's own words, e.g. 'every night at 11 pm', "
                        "'tomorrow at 7 am', 'in 30 minutes'",
            }

        nxt = nlu.compute_next(rule, self.now)
        if nxt is None:
            return {"error": "that time is in the past"}

        job_id = db.add_job(
            device_id=d["id"],
            device_name=d["name"],
            actions=clean,
            rule=rule,
            next_run_at=nxt.isoformat(timespec="seconds"),
            repeat_label=label,
            source_text=str(when_text or ""),
        )
        self.actions.extend({**a, "device_id": d["id"], "device_name": d["name"]}
                            for a in clean)
        self.targets.append({"id": d["id"], "name": d["name"], "actions": clean,
                             "scheduled": job_id})
        db.log("tool.schedule", f"{d['name']} {clean} {rule} -> job {job_id}")
        return {
            "ok": True,
            "job_id": job_id,
            "device": d["name"],
            "repeat": label or "once",
            "next_run": nxt.strftime("%b %d, %H:%M"),
        }

    def _t_ask_user(self, question: str = "") -> dict:
        self.question = clean_reply(question) or "Which device do you mean?"
        self.finished = True
        return {"ok": True, "asked": self.question}

    def _t_finish(self, reply: str = "") -> dict:
        self.reply = clean_reply(reply) or None
        self.finished = True
        return {"ok": True}

    # ------------------------------------------------ result

    def summary(self) -> str:
        """What to say when the model stopped without a `finish` call.

        Phrased per device. A whole-home request produces one identical action
        per room, and flattening those gave "turn off, turn off, turn off" —
        which names neither the devices nor the rooms.
        """
        if not self.targets:
            return ""

        names = [t["name"] for t in self.targets]
        plans = [t.get("actions") or [] for t in self.targets]

        # the usual whole-home case: one shared action, so say it once
        if plans[0] and all(p == plans[0] for p in plans):
            did = ", ".join(_short(a) for a in plans[0])
            return f"Done \u2014 {_and(names)}: {did}."

        parts = []
        for name, acts in zip(names, plans):
            tail = ", ".join(_short(a) for a in acts) if acts else "done"
            parts.append(f"{name} {tail}")
        return "Done \u2014 " + "; ".join(parts) + "."


def _short(a: dict) -> str:
    """One action as a noun-ish fragment: "off", "24°C", "cool", "fan low"."""
    name, value = a.get("action"), a.get("value")
    if name == "power":
        return "on" if value else "off"
    if name == "temperature":
        return f"{value}\u00b0C"
    if name == "mode":
        return MODE_LABELS.get(value, str(value))
    if name == "fan":
        return f"fan {FAN_LABELS.get(value, value)}"
    return f"{name}={value}"


def _and(items: list[str]) -> str:
    """'a, b and c' — a comma before the last name reads wrong in a sentence."""
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]

