"""Agent loop: understand -> decide -> execute -> reply.

Two boundaries matter here:

1. Execution stays in deterministic code. The LLM only translates language into
   a structured intent; it never touches a device.
2. Understanding is stateful. A request that is missing a slot ("it's stuffy")
   is remembered and completed by the next turn instead of being thrown away —
   see `_complete_pending`.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from . import db, llm, nlu
from .devices.base import FAN_LABELS, MODE_LABELS, describe_action

# how long a half-finished request stays open
PENDING_TTL = 120

CANCEL_PENDING_RE = re.compile(
    r"\b(cancel|never\s*mind|forget\s*it|nothing|drop\s*it)\b", re.I)


class Agent:
    def __init__(self, adapter, cfg: dict) -> None:
        self.adapter = adapter
        self.cfg = cfg
        # remembers the last device mentioned, so "set it to 24" works
        self._last_device_id: str | None = None
        # a request waiting for a missing slot
        self._pending: dict | None = None

    # ------------------------------------------------ entry point

    def handle(self, text: str) -> dict:
        now = datetime.now()
        devices = self.adapter.list_devices()
        default_room = self.cfg.get("agent", {}).get("default_room", "Living Room")

        fallback = None
        if self._last_device_id:
            fallback = next((d for d in devices if d["id"] == self._last_device_id), None)

        # 1. finish what we were asking about, if anything
        intent = self._complete_pending(text, devices, default_room, now)

        # 2. otherwise parse normally
        if intent is None:
            intent = nlu.parse(text, devices, now, default_room, fallback)

            # Rules first. Only hand the long tail to the LLM.
            if intent["intent"] == "unknown" and llm.available(self.cfg):
                alt = llm.plan(self.cfg, text, devices, now, default_room)
                if alt and alt["intent"] != "unknown":
                    alt["clarify"] = alt.get("clarify") or intent.get("clarify")
                    intent = alt

            # remember a request that is only missing one slot
            if intent["intent"].startswith("need_"):
                self._pending = {
                    "missing": intent["missing"],
                    "partial": intent,
                    "expires": now + timedelta(seconds=PENDING_TTL),
                }

        result = self._execute(intent, now)
        result["intent_name"] = intent["intent"]
        result["raw_text"] = text

        targets = self._targets(intent)
        if targets:
            self._last_device_id = targets[0]["id"]

        db.add_message("user", text, {"intent": intent["intent"]})
        db.add_message("assistant", result["reply"], {
            "intent": intent["intent"],
            "actions": result.get("actions") or [],
        })
        return result

    # ------------------------------------------------ slot filling

    def _complete_pending(self, text: str, devices: list[dict],
                          default_room: str, now: datetime) -> dict | None:
        p = self._pending
        if not p:
            return None
        if now > p["expires"]:
            self._pending = None
            return None

        if CANCEL_PENDING_RE.search(text):
            self._pending = None
            return {"intent": "dropped", "actions": [], "targets": [],
                    "device_id": None, "device_name": None, "rule": None,
                    "repeat_label": "", "confidence": 1.0, "raw_text": text,
                    "clarify": "Okay, dropped that."}

        # If the new message is itself a complete request, it is not an answer
        # to our question — drop the pending one instead of hijacking it.
        probe = nlu.parse(text, devices, now, default_room, None)
        if probe["intent"] in ("control", "schedule", "query", "list", "cancel"):
            self._pending = None
            return None

        partial = dict(p["partial"])
        partial.pop("missing", None)
        partial["clarify"] = None  # the question has been answered

        if p["missing"] == "target":
            targets, label = nlu.resolve_targets(text, devices, default_room, None)
            if not targets:
                return None  # still unclear; let the normal parser try
            nlu.set_targets(partial, targets, label)
        else:  # missing action
            acts = nlu.extract_actions(text) or nlu.extract_comfort(text)
            if not acts:
                return None
            partial["actions"] = acts

        partial["intent"] = "schedule" if partial.get("rule") else "control"
        partial["confidence"] = 0.9
        self._pending = None
        return partial

    # ------------------------------------------------ execution

    @staticmethod
    def _targets(it: dict) -> list[dict]:
        if it.get("targets"):
            return it["targets"]
        if it.get("device_id"):
            return [{"id": it["device_id"], "name": it.get("device_name") or it["device_id"]}]
        return []

    def _execute(self, it: dict, now: datetime) -> dict:
        kind = it["intent"]
        if kind == "query":
            return self._do_query(it)
        if kind == "control":
            return self._do_control(it)
        if kind == "schedule":
            return self._do_schedule(it, now)
        if kind == "cancel":
            return self._do_cancel(it)
        if kind == "list":
            return self._do_list()
        if kind == "dropped":
            return {"reply": it.get("clarify") or "Okay.", "ok": True, "actions": []}
        if kind.startswith("need_"):
            return {"reply": it.get("clarify") or "Could you be more specific?",
                    "ok": False, "actions": [], "awaiting": it.get("missing")}
        return {
            "reply": it.get("clarify") or "Sorry, I did not catch that.",
            "ok": False,
            "actions": [],
        }

    def _do_control(self, it: dict) -> dict:
        targets = self._targets(it)
        if not targets:
            return {"reply": "I could not find that device.", "ok": False, "actions": []}
        if len(targets) == 1:
            return self._do_control_one(it, targets[0])

        outcomes = []
        for t in targets:
            device = self.adapter.get_device(t["id"])
            if not device:
                outcomes.append((t["name"], False, "device not found"))
                continue
            if not device.get("online", True):
                outcomes.append((device["name"], False, "offline"))
                continue
            res = self.adapter.apply(device["id"], it["actions"])
            db.log("control", f"{device['name']} {it['actions']} -> {res.get('ok')}")
            outcomes.append((device["name"], bool(res.get("ok")), res.get("error")))

        ok_count = sum(1 for o in outcomes if o[1])
        lines = [f"Done \u2014 {ok_count} of {len(outcomes)} devices updated:"]
        for name, good, err in outcomes:
            if good:
                lines.append(f"\u00b7 {name} \u2014 {self.action_summary(it['actions'])}")
            else:
                lines.append(f"\u00b7 {name} \u2014 skipped: {self._strip_name(err, name)}")
        if it.get("clarify"):
            lines.append(it["clarify"])
        return {"reply": "\n".join(lines), "ok": ok_count > 0, "actions": it["actions"]}

    @staticmethod
    def _strip_name(err: str | None, name: str) -> str:
        """Adapters word errors as "<Device> does not support X".

        In a per-device list the name is already the line prefix, so repeating it
        reads as a stutter: "· Purifier — skipped: Purifier does not support…".
        """
        err = err or "command rejected"
        if err.startswith(name):
            rest = err[len(name):].lstrip(" :-\u2014")
            if rest:
                return rest
        return err

    def _do_control_one(self, it: dict, target: dict) -> dict:
        device = self.adapter.get_device(target["id"])
        if not device:
            return {"reply": "I could not find that device.", "ok": False, "actions": []}
        if not device.get("online", True):
            return {"reply": f"{device['name']} is offline, so nothing was sent.",
                    "ok": False, "actions": []}

        res = self.adapter.apply(device["id"], it["actions"])
        db.log("control", f"{device['name']} {it['actions']} -> {res.get('ok')}")

        if not res.get("ok"):
            return {"reply": f"Command failed: {res.get('error')}", "ok": False, "actions": []}

        state = self.adapter.get_state(device["id"])
        power_act = next((a for a in it["actions"] if a["action"] == "power"), None)

        if power_act:
            reply = f"Done \u2014 {device['name']} is {'on' if power_act['value'] else 'off'}."
            if state.get("power"):
                brief = self.brief(device, state)
                if brief:
                    reply += f"\nCurrently: {brief}."
        else:
            reply = f"Done \u2014 {self.phrase(device['name'], it['actions'])}."

        if res.get("dry_run"):
            reply += "\n(dry-run mode: payload logged, not sent)"
        if it.get("clarify"):
            reply += "\n" + it["clarify"]
        return {"reply": reply, "ok": True, "actions": it["actions"]}

    def _do_schedule(self, it: dict, now: datetime) -> dict:
        targets = self._targets(it)
        if not targets:
            return {"reply": "I could not find that device, so nothing was scheduled.",
                    "ok": False, "actions": []}

        rule = it["rule"]
        nxt = nlu.compute_next(rule, now)
        if nxt is None:
            return {"reply": "I could not work out that time. Try another phrasing?",
                    "ok": False, "actions": []}

        label = it.get("repeat_label") or "one time"
        created = []
        for t in targets:
            job_id = db.add_job(
                device_id=t["id"], device_name=t["name"],
                actions=it["actions"], rule=rule,
                next_run_at=nxt.isoformat(timespec="seconds"),
                repeat_label=it.get("repeat_label", ""),
                source_text=it.get("raw_text", ""),
            )
            created.append(job_id)
        db.log("schedule.create", f"{created} {[t['id'] for t in targets]} "
                                  f"{it['actions']} {rule}")

        when = nxt.strftime("%b %d, %H:%M")
        if len(targets) == 1:
            reply = (f"Scheduled \u2014 {label}: "
                     f"{self.phrase(targets[0]['name'], it['actions'])}.\n"
                     f"Next run: {when} (job {created[0]})")
        else:
            lines = [f"Scheduled \u2014 {label}, {len(created)} tasks:"]
            for t in targets:
                lines.append(f"\u00b7 {self.phrase(t['name'], it['actions'])}")
            lines.append(f"Next run: {when}")
            reply = "\n".join(lines)

        if it.get("clarify"):
            reply += "\n" + it["clarify"]
        return {"reply": reply, "ok": True, "actions": it["actions"],
                "jobs": created,
                "job": {"id": created[0],
                        "next_run_at": nxt.isoformat(timespec="seconds")}}

    def _do_cancel(self, it: dict) -> dict:
        jobs = db.list_jobs(only_enabled=True)
        if not jobs:
            return {"reply": "There are no active scheduled tasks.", "ok": True, "actions": []}

        if it.get("cancel_all"):
            n = db.clear_jobs()
            db.log("schedule.clear", str(n))
            return {"reply": f"Cleared {n} scheduled task(s).", "ok": True, "actions": []}

        device, _ = nlu.match_device(it.get("raw_text", ""), self.adapter.list_devices())
        if device:
            hit = [j for j in jobs if j["device_id"] == device["id"]]
            if not hit:
                return {"reply": f"There is nothing scheduled for {device['name']}.",
                        "ok": True, "actions": []}
            for j in hit:
                db.delete_job(j["id"])
            return {"reply": f"Cancelled {len(hit)} task(s) for {device['name']}.",
                    "ok": True, "actions": []}

        if len(jobs) == 1:
            db.delete_job(jobs[0]["id"])
            return {"reply": f"Cancelled: {jobs[0]['repeat_label']} "
                             f"{jobs[0]['device_name']}.", "ok": True, "actions": []}

        lines = [f"You have {len(jobs)} scheduled tasks. Which one should I cancel?"]
        for j in jobs:
            lines.append(f"\u00b7 {j['repeat_label'] or 'one time'} "
                         f"{j['device_name']} ({j['next_run_at'][5:16].replace('T', ' ')})")
        lines.append('Say "cancel all" to clear everything.')
        return {"reply": "\n".join(lines), "ok": False, "actions": []}

    def _do_list(self) -> dict:
        jobs = db.list_jobs(only_enabled=True)
        if not jobs:
            return {"reply": "No scheduled tasks yet. Try \"turn off the living room "
                             "AC every night at 11 pm\".", "ok": True, "actions": []}
        lines = [f"{len(jobs)} scheduled task(s):"]
        for j in jobs:
            nxt = j["next_run_at"][5:16].replace("T", " ")
            lines.append(f"\u00b7 {j['repeat_label'] or 'one time'} \u2014 "
                         f"{self.phrase(j['device_name'], j['actions'])} \u2014 next {nxt}")
        return {"reply": "\n".join(lines), "ok": True, "actions": []}

    def _do_query(self, it: dict) -> dict:
        targets = self._targets(it)
        device = self.adapter.get_device(targets[0]["id"]) if targets else None
        if not device:
            return {"reply": it.get("clarify") or "Which device should I check?",
                    "ok": False, "actions": []}
        state = self.adapter.get_state(device["id"])
        reply = self.describe(device, state)
        if it.get("clarify"):
            reply += "\n" + it["clarify"]
        return {"reply": reply, "ok": True, "actions": [], "state": state}

    # ------------------------------------------------ phrasing

    @staticmethod
    def action_summary(actions: list[dict]) -> str:
        return ", ".join(describe_action(a) for a in actions)

    @staticmethod
    @staticmethod
    def _phrase_one(device_name: str, name, value, with_name: bool = True) -> str:
        """One action as a clause. `with_name` is off for every action after the
        first, so a two-action command does not repeat the device name."""
        if name == "power":
            verb = "on" if value else "off"
            return f"turn {verb} the {device_name}" if with_name else f"turn {verb}"
        if name == "temperature":
            return (f"set the {device_name} to {value}\u00b0C" if with_name
                    else f"{value}\u00b0C")
        if name == "mode":
            label = MODE_LABELS.get(value, value)
            return (f"switch the {device_name} to {label}" if with_name
                    else f"switch to {label}")
        if name == "fan":
            label = FAN_LABELS.get(value, value)
            return (f"set the {device_name} fan to {label}" if with_name
                    else f"fan {label}")
        return f"{name}={value}"

    @staticmethod
    def phrase(device_name: str, actions: list[dict]) -> str:
        """Turn an action list into a readable phrase, e.g.
        'turn off the Living Room AC' or
        'set the Living Room AC to 24°C, fan low'.

        The device name appears once. Repeating it per verb produced replies
        like "switch the AC to heating and set the AC fan to low", which reads
        as a stutter rather than a sentence.
        """
        if not actions:
            return f"nothing to the {device_name}"

        first = actions[0]
        parts = [Agent._phrase_one(device_name, first.get("action"),
                                   first.get("value"), with_name=True)]
        parts += [Agent._phrase_one(device_name, a.get("action"), a.get("value"),
                                    with_name=False) for a in actions[1:]]
        return ", ".join(parts)

    @staticmethod
    def brief(device: dict, state: dict) -> str:
        """One-line state summary without the device name."""
        if not device.get("online", True) or state.get("power") is None:
            return ""
        t = device["type"]
        if t == "air_conditioner":
            if state.get("power"):
                bits = []
                mode_txt = (MODE_LABELS.get(state["mode"], state["mode"])
                            if state.get("mode") else "")
                temp_txt = f"{state['target_temp']}\u00b0C" if state.get("target_temp") else ""
                if mode_txt and temp_txt:
                    bits.append(f"{mode_txt} {temp_txt}")
                elif mode_txt or temp_txt:
                    bits.append(mode_txt or temp_txt)
                if state.get("fan"):
                    bits.append(f"fan {FAN_LABELS.get(state['fan'], state['fan'])}")
                if state.get("current_temp") is not None:
                    bits.append(f"room {state['current_temp']}\u00b0C")
                return ", ".join(bits)
            if state.get("current_temp") is not None:
                return f"off, room {state['current_temp']}\u00b0C"
            return "off"
        if t == "washer":
            if state.get("power"):
                return f"washing, about {state.get('remaining_min', '?')} min left"
            return "standby"
        if t == "refrigerator":
            return (f"fridge {state.get('fridge_temp', '-')}\u00b0C, "
                    f"freezer {state.get('freezer_temp', '-')}\u00b0C")
        if t == "air_purifier":
            return f"PM2.5 {state.get('pm25', '-')}"
        return "running" if state.get("power") else "off"

    @staticmethod
    def describe(device: dict, state: dict) -> str:
        name = device["name"]
        if not device.get("online", True):
            return f"{name} is offline, so I cannot read its state."
        power = state.get("power")
        if power is None:
            return f"I could not read the state of {name}."

        if device["type"] == "air_conditioner":
            if power:
                detail = Agent.brief(device, state)
                return f"{name} is running \u2014 {detail}." if detail else f"{name} is running."
            tail = (f" Room {state['current_temp']}\u00b0C."
                    if state.get("current_temp") is not None else "")
            return f"{name} is off.{tail}"

        if device["type"] == "washer":
            if power:
                left = (f", about {state['remaining_min']} min left"
                        if state.get("remaining_min") else "")
                return f"{name} is washing{left}."
            return f"{name} is on standby."

        if device["type"] == "refrigerator":
            return (f"{name} is running \u2014 fridge {state.get('fridge_temp', '-')}\u00b0C, "
                    f"freezer {state.get('freezer_temp', '-')}\u00b0C.")

        if device["type"] == "air_purifier":
            if power:
                return f"{name} is running, PM2.5 {state.get('pm25', '-')}."
            return f"{name} is off, PM2.5 {state.get('pm25', '-')}."

        return f"{name} is {'running' if power else 'off'}."

    # ------------------------------------------------ snapshot for the UI

    def snapshot(self) -> dict:
        devices = []
        for d in self.adapter.list_devices():
            try:
                state = self.adapter.get_state(d["id"])
            except Exception:
                state = {"power": None}
            devices.append({**d, "state": state, "summary": self.describe(d, state),
                            "brief": self.brief(d, state)})
        jobs = db.list_jobs()
        for j in jobs:
            j["summary"] = self.phrase(j["device_name"], j["actions"])
        return {
            "devices": devices,
            "jobs": jobs,
            "backend": getattr(self.adapter, "backend_name", "unknown"),
            "llm_enabled": llm.available(self.cfg),
            "awaiting": (self._pending or {}).get("missing"),
            "now": datetime.now().isoformat(timespec="seconds"),
        }
