"""Agent loop: understand -> decide -> execute -> reply.

The important boundary: execution stays in deterministic code. The LLM only
translates language into a structured intent; it never touches a device.
"""
from __future__ import annotations

from datetime import datetime

from . import db, llm, nlu
from .devices.base import FAN_LABELS, MODE_LABELS, describe_action


class Agent:
    def __init__(self, adapter, cfg: dict) -> None:
        self.adapter = adapter
        self.cfg = cfg
        # remembers the last device mentioned, so "set it to 24" works
        self._last_device_id: str | None = None

    # ------------------------------------------------ entry point

    def handle(self, text: str) -> dict:
        now = datetime.now()
        devices = self.adapter.list_devices()
        default_room = self.cfg.get("agent", {}).get("default_room", "Living Room")

        fallback = None
        if self._last_device_id:
            fallback = next((d for d in devices if d["id"] == self._last_device_id), None)

        intent = nlu.parse(text, devices, now, default_room, fallback)

        # Rules first. Only hand the long tail to the LLM.
        if intent["intent"] == "unknown" and llm.available(self.cfg):
            alt = llm.plan(self.cfg, text, devices, now, default_room)
            if alt and alt["intent"] != "unknown":
                alt["clarify"] = alt.get("clarify") or intent.get("clarify")
                intent = alt

        result = self._execute(intent, now)
        result["intent_name"] = intent["intent"]
        result["raw_text"] = text

        if intent.get("device_id"):
            self._last_device_id = intent["device_id"]

        db.add_message("user", text, {"intent": intent["intent"]})
        db.add_message("assistant", result["reply"], {
            "intent": intent["intent"],
            "actions": result.get("actions") or [],
        })
        return result

    # ------------------------------------------------ execution

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
        return {
            "reply": it.get("clarify") or "Sorry, I did not catch that.",
            "ok": False,
            "actions": [],
        }

    def _do_control(self, it: dict) -> dict:
        device = self.adapter.get_device(it["device_id"])
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
            rest = [describe_action(a) for a in it["actions"] if a["action"] != "power"]
            if rest:
                reply = reply[:-1] + ", " + ", ".join(rest) + "."
            if state.get("power"):
                brief = self.brief(device, state)
                if brief:
                    reply += f"\nCurrently: {brief}."
        else:
            done = ", ".join(describe_action(a) for a in it["actions"])
            reply = f"Done \u2014 {device['name']} {done}."
            brief = self.brief(device, state)
            if brief:
                reply += f"\nCurrently: {brief}."

        if res.get("dry_run"):
            reply += "\n(dry-run mode: payload logged, not sent)"
        if it.get("clarify"):
            reply += "\n" + it["clarify"]
        return {"reply": reply, "ok": True, "actions": it["actions"]}

    def _do_schedule(self, it: dict, now: datetime) -> dict:
        device = self.adapter.get_device(it["device_id"])
        if not device:
            return {"reply": "I could not find that device, so nothing was scheduled.",
                    "ok": False, "actions": []}

        rule = it["rule"]
        nxt = nlu.compute_next(rule, now)
        if nxt is None:
            return {"reply": "I could not work out that time. Try another phrasing?",
                    "ok": False, "actions": []}

        job_id = db.add_job(
            device_id=device["id"], device_name=device["name"],
            actions=it["actions"], rule=rule,
            next_run_at=nxt.isoformat(timespec="seconds"),
            repeat_label=it.get("repeat_label", ""), source_text=it.get("raw_text", ""),
        )
        db.log("schedule.create", f"{job_id} {device['name']} {it['actions']} {rule}")

        label = it.get("repeat_label") or "one time"
        reply = (f"Scheduled \u2014 {label}: {self.phrase(device['name'], it['actions'])}.\n"
                 f"Next run: {nxt.strftime('%b %d, %H:%M')} (job {job_id})")
        if it.get("clarify"):
            reply += "\n" + it["clarify"]
        return {"reply": reply, "ok": True, "actions": it["actions"],
                "job": {"id": job_id, "next_run_at": nxt.isoformat(timespec="seconds")}}

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
        device = self.adapter.get_device(it["device_id"]) if it.get("device_id") else None
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
    def phrase(device_name: str, actions: list[dict]) -> str:
        """Turn an action list into a readable phrase, e.g.
        'turn off the Living Room AC and set it to 24°C'."""
        parts = []
        for a in actions:
            n, v = a.get("action"), a.get("value")
            if n == "power":
                parts.append(f"turn {'on' if v else 'off'} the {device_name}")
            elif n == "temperature":
                parts.append(f"set the {device_name} to {v}\u00b0C")
            elif n == "mode":
                parts.append(f"switch the {device_name} to {MODE_LABELS.get(v, v)}")
            elif n == "fan":
                parts.append(f"set the {device_name} fan to {FAN_LABELS.get(v, v)}")
            else:
                parts.append(f"{n}={v}")
        return " and ".join(parts)

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
                bits = [f"{name} is running"]
                detail = Agent.brief(device, state)
                if detail:
                    bits.append(detail)
                return " \u2014 ".join(bits) + "."
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
            "now": datetime.now().isoformat(timespec="seconds"),
        }
