"""Proactive decision engine (Phase 2).

The loop, once a second:

    sensors.tick -> risk.evaluate -> policy.candidates -> DECIDE -> execute / ask

Where the LLM sits in that chain is the whole design. It is not the timer, not
the sensor, and not the safety gate — those are deterministic code, and Phase 1
already argued why a model must never be any of them. The LLM owns exactly the
decisions rules cannot express:

  * which of the candidate actions to take, and which to leave alone
  * in what order, when a mold risk and a thermal risk fire together
  * how to phrase a Tier-2 question so a senior can answer it in one word

That is a real decision, not polishing: given the same candidates the model may
drop one because the dehumidifier is offline, or abstain entirely because the
trend says the drum is already drying. Its output is validated against a
whitelist before anything moves — it can be wrong about *what* to do, but it
cannot corrupt the house, and a parse failure falls back to the safe default
(Tier 1 executes, Tier 2 is only ever proposed).

Tier-2 execution goes through ApprovalQueue.confirm(), and only the user's
"yes" gets there.
"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime

from . import approval, db, llm, policy, risk, sensors
from .devices.base import describe_action

DECIDE_SYSTEM = """You are the decision layer of a proactive smart-home agent for a
household that includes a senior. Sensors have flagged a risk; rules have
proposed candidate actions. Your job is to choose.

Choose only from the candidates. For each chosen candidate give its exact kind.
You may:
- take some and leave others (a candidate whose device is offline or already in
  the wanted state should be left alone),
- reorder them when the order matters,
- take none, if the evidence says the situation is resolving on its own.

You cannot execute anything yourself. Tier-1 actions you choose run
automatically; Tier-2 actions you choose are put to the user for approval and
never run without a yes.

Answer with JSON only:
{"chosen": [{"kind": "candidate_kind", "reason": "one short clause"}],
 "note": "one sentence for the household log, plain language, no jargon"}"""

QUESTION_SYSTEM = """You are writing one short question for an elderly homeowner.
The house wants to take an action that changes their plans, so it must ask
first. Rules:
- one question, at most two short sentences, no technical terms
- name the appliance and what will change, and the deadline for answering
- offer a one-word reply (yes / no)
- warm and plain, never alarming
Output the question text only."""


def _clean(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    return text


class ProactiveEngine:
    """Watches the simulated house and intervenes under the risk-graded policy."""

    def __init__(self, adapter, cfg: dict) -> None:
        pcfg = cfg.get("proactive", {}) or {}
        self.adapter = adapter
        self.cfg = cfg
        self.tick_seconds = float(pcfg.get("tick_seconds", 1.0))
        self.cooldown_seconds = float(pcfg.get("cooldown_seconds", 180))
        self.llm_decide = bool(pcfg.get("llm_decide", True)) and llm.available(cfg)

        self.sensors = sensors.SensorSimulator(tick_seconds=self.tick_seconds)
        self.approvals = approval.ApprovalQueue(
            ttl_seconds=policy.APPROVAL_TTL_SECONDS)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cooldown: dict[str, float] = {}     # risk kind -> monotonic deadline
        self._active: dict[str, dict] = {}        # risk kind -> last intervention
        self._last_snapshot: dict = {}

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.sensors.start()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="proactive")
        self._thread.start()
        db.log("proactive.start",
               f"engine started (llm_decide={'on' if self.llm_decide else 'off'})")

    def stop(self) -> None:
        self._stop.set()
        self.sensors.stop()

    def _loop(self) -> None:
        while not self._stop.wait(self.tick_seconds):
            try:
                self.tick()
            except Exception as exc:
                # The engine watches the house; it must not be able to break it.
                db.log("proactive.error", f"{type(exc).__name__}: {exc}", "error")

    # ---------------------------------------------------------------- one pass

    def tick(self, now: datetime | None = None) -> dict:
        """One observe-decide-act pass. Returns what happened (for tests/demos)."""
        now = now or datetime.now()
        snap = self.sensors.snapshot()
        self.sensors.record(snap)
        self._last_snapshot = snap

        fired = [r for r in risk.evaluate(snap, self.sensors.history, now)
                 if r.score >= r.threshold]
        events: list[dict] = []

        pending_kinds = {a.risk_kind for a in self.approvals.pending()}
        for r in fired:
            if self._cooling(r.kind):
                continue
            if r.kind in pending_kinds:
                continue  # already waiting on the user for this risk
            events.append(self._intervene(r, snap, now))

        for ap in self.approvals.expire_due(now):
            events.append({"type": "expired", "approval": ap.id})
            db.add_message("system",
                           f"\u23f3 No answer on: {ap.summary} "
                           f"\u2014 nothing was changed. The house stays as it was.",
                           {"approval": ap.to_dict()})
        return {"ts": snap["ts"], "risks": [r.kind for r in fired], "events": events}

    # ---------------------------------------------------------------- intervene

    def _intervene(self, r: risk.Risk, snap: dict, now: datetime) -> dict:
        cands = policy.candidates(r, snap)
        if not cands:
            return {"type": "no_candidates", "risk": r.kind}

        chosen, decided_by = self._decide(r, cands, snap)
        tier1 = [c for c in chosen if policy.tier_of(c.kind) == policy.TIER1]
        tier2 = [c for c in chosen if policy.tier_of(c.kind) == policy.TIER2]

        executed = []
        for c in tier1:
            res = self.adapter.apply(c.device_id, c.actions)
            self.sensors.note_action(c.device_id, c.actions)
            ok = bool(res.get("ok"))
            executed.append({"kind": c.kind, "label": c.label, "ok": ok,
                             "error": res.get("error")})
            db.log("proactive.tier1",
                   f"{r.kind}: {c.label} -> {'ok' if ok else res.get('error')}")

        approval_id = None
        if tier2:
            question = self._phrase(r, tier2, snap)
            ap = self.approvals.propose(
                risk_kind=r.kind,
                question=question,
                summary="; ".join(c.label for c in tier2),
                actions=[_candidate_dict(c) for c in tier2],
            )
            approval_id = ap.id
            db.add_message("system", question, {"approval": ap.to_dict()})

        labels = [c.label for c in tier1]
        head = r.headline
        if labels:
            body = f"\U0001f6e1 {head}. I " + " and ".join(labels) + "."
        elif tier2:
            body = f"\U0001f6e1 {head}. This one changes your plans, so I am asking first."
        else:
            body = f"\U0001f6e1 {head}. Checked the candidates \u2014 nothing to do right now."
        db.add_message("system", body, {
            "risk": r.kind, "score": round(r.score, 1), "drivers": r.drivers,
            "decided_by": decided_by, "approval": approval_id,
            "executed": executed,
        })

        self._cooldown[r.kind] = time.monotonic() + self.cooldown_seconds
        self._active[r.kind] = {"score": r.score, "at": now.isoformat(timespec="seconds"),
                                "executed": [c.kind for c in tier1],
                                "approval": approval_id}
        return {"type": "intervened", "risk": r.kind, "decided_by": decided_by,
                "tier1": [c.kind for c in tier1], "tier2": [c.kind for c in tier2],
                "approval": approval_id}

    # ---------------------------------------------------------------- the LLM's job

    def _decide(self, r: risk.Risk, cands: list, snap: dict):
        """Choose among candidates. Returns (chosen_in_order, decided_by)."""
        if not self.llm_decide:
            return list(cands), "rules"
        user = (
            f"Risk: {r.headline}\n"
            f"Drivers: {json.dumps(r.drivers, ensure_ascii=False)}\n"
            f"User home: {snap['occupancy'].get('user_home')}\n\n"
            "Candidates:\n" + "\n".join(
                f"- kind={c.kind} tier={policy.tier_of(c.kind)} "
                f"device={c.device_name} does: {c.label} (why: {c.reason})"
                for c in cands)
        )
        try:
            msg = llm.chat(self.cfg, [
                {"role": "system", "content": DECIDE_SYSTEM},
                {"role": "user", "content": user},
            ], json_mode=True, temperature=0, max_tokens=400)
            raw = json.loads(_clean(msg.get("content") or ""))
            kinds = [c.get("kind") for c in (raw.get("chosen") or [])]
        except Exception as exc:
            db.log("proactive.decide_fallback",
                   f"{type(exc).__name__}: {exc}; took the rule default")
            return list(cands), "rules"

        by_kind = {c.kind: c for c in cands}
        chosen, seen = [], set()
        for k in kinds:
            if k in by_kind and k not in seen:
                chosen.append(by_kind[k])
                seen.add(k)
        if not chosen and not raw.get("note"):
            # Empty with no explanation is a parse we do not trust; the safe
            # default still runs Tier 1 and only proposes Tier 2.
            db.log("proactive.decide_fallback", "empty decision; took the rule default")
            return list(cands), "rules"
        note = str(raw.get("note") or "").strip()
        if note:
            db.log("proactive.decide", f"{r.kind}: {note}")
        return chosen, "llm"

    def _phrase(self, r: risk.Risk, tier2: list, snap: dict) -> str:
        """One plain-language question for the Tier-2 request."""
        labels = "; ".join(c.label for c in tier2)
        fallback = (f"{r.headline}. I would like to {labels}. "
                    f"Shall I? Reply yes or no \u2014 no answer in 5 minutes "
                    f"keeps everything as it is.")
        if not llm.available(self.cfg):
            return fallback
        try:
            msg = llm.chat(self.cfg, [
                {"role": "system", "content": QUESTION_SYSTEM},
                {"role": "user", "content":
                    f"Situation: {r.headline}. Requested change: {labels}. "
                    f"User is {'home' if snap['occupancy'].get('user_home') else 'out'}."},
            ], temperature=0.2, max_tokens=120)
            text = (msg.get("content") or "").strip()
            if text and len(text) < 400:
                return text
        except Exception:
            pass
        return fallback

    # ---------------------------------------------------------------- user decisions

    def confirm(self, approval_id: str) -> dict:
        """The user said yes. Execute the Tier-2 actions now."""
        ap = self.approvals.confirm(approval_id)
        if not ap:
            return {"ok": False, "error": "no pending approval with that id"}
        executed = []
        for item in ap.actions:
            res = self.adapter.apply(item["device_id"], item["actions"])
            self.sensors.note_action(item["device_id"], item["actions"])
            executed.append({"device_id": item["device_id"], "ok": bool(res.get("ok")),
                             "error": res.get("error")})
            job = item.get("job")
            if job and res.get("ok"):
                db.add_job(job["device_id"], item.get("device_name", job["device_id"]),
                           job["actions"], job["rule"], job["rule"]["run_at"],
                           repeat_label=job.get("repeat_label", ""),
                           source_text=f"approved:{approval_id}")
        done = ", ".join(describe_action(a) for it in ap.actions for a in it["actions"])
        db.add_message("system",
                       f"\u2713 Approved \u2014 done: {done}. "
                       + ("A job was scheduled for the restart."
                          if any(it.get("job") for it in ap.actions) else ""),
                       {"approval": ap.to_dict(), "executed": executed})
        return {"ok": True, "approval": ap.to_dict(), "executed": executed}

    def decline(self, approval_id: str) -> dict:
        ap = self.approvals.decline(approval_id)
        if not ap:
            return {"ok": False, "error": "no pending approval with that id"}
        db.add_message("system",
                       f"\u2717 Declined \u2014 nothing was changed.",
                       {"approval": ap.to_dict()})
        return {"ok": True, "approval": ap.to_dict()}

    # ---------------------------------------------------------------- state

    def _cooling(self, kind: str) -> bool:
        until = self._cooldown.get(kind)
        return until is not None and time.monotonic() < until

    def state(self) -> dict:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "llm_decide": self.llm_decide,
            "scenario": self._last_snapshot.get("scenario"),
            "rooms": self._last_snapshot.get("rooms"),
            "appliances": self._last_snapshot.get("appliances"),
            "occupancy": self._last_snapshot.get("occupancy"),
            "active": self._active,
            "pending_approvals": [a.to_dict() for a in self.approvals.pending()],
        }


def _candidate_dict(c) -> dict:
    return {"kind": c.kind, "device_id": c.device_id, "device_name": c.device_name,
            "actions": c.actions, "label": c.label, "job": c.job}
