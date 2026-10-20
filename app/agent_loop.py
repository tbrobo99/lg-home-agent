"""Tier 3: a bounded tool loop for sentences the rules cannot parse.

The cascade in `agent.py` has three tiers, each entered only when the previous
one gives up:

    1. rules (nlu.py)        deterministic, 100% on the benchmark, ~1 ms
    2. single-shot LLM       one call, still returns an intent dict, ~2 s
    3. this loop             up to 8 calls, calls tools, ~2 s per step

Tier 3 is the only place a model is allowed to *act*, and it is allowed to act
only through `tools.ToolBox`, which validates everything and never accepts a
timestamp. So the loop can be wrong about what to do, but it cannot corrupt the
job table or send a malformed command.

It is also the only tier that can recover from its own mistakes: a bad device id
comes back as an observation, and the next step sees it.

**Agent mode.** The cascade is a workflow: the tier that answers a sentence is
chosen by the shape of the sentence in `_escalate`, in code, before any model
sees it. Agent mode keeps the same loop but inverts the relationship for
requests the rules *did* understand: the parse is handed down as an *offer*
(`adopt_parse`), and the model either accepts it in one step or plans for
itself. The fast path survives as a shortcut the model chooses to take, not as
a branch a developer wrote. That is the difference between a workflow with a
model in it and an agent that can be told it is wrong.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

from . import db, llm, nlu
from .tools import TOOL_SCHEMAS, ToolBox, clean_reply

MAX_STEPS = 8
DEADLINE_SECONDS = 25.0

SYSTEM_PROMPT = """You are the fallback layer of a smart-home assistant. The rule
engine could not understand the user's message, so you have tools. Use them.

Current time: {now}
Default room: {default_room}
Devices:
{devices}

Rules:
1. Call `list_devices` or `get_state` before acting if the target is unclear.
2. One `control` call per device, but you may emit several calls in one turn.
   For a whole-home request, emit one `control` per device together. Devices
   cannot be created or renamed.
3. If the user gives any time — "at 11 pm", "in 20 minutes", "tomorrow", "every
   night" — the work happens THEN, so call `schedule`, never `control`. Pass the
   time as the user said it; never convert it to a clock time yourself, because
   a deterministic parser does that and it will reject what it cannot read.
4. If the request is genuinely ambiguous, call `ask_user` with one short question.
   A whole-home request is not ambiguous: see rule 8.
5. When the work is done, call `finish` with one short sentence.
6. Never claim something happened unless a tool said it did.
7. If the user describes a problem and one device plainly fixes it, fix it. Do
   not offer to: "the air in here is disgusting" is a request, not a question.
8. "The flat", "the house", "the whole place", "the home", "everything", "the
   lot" means EVERY device. Call `list_devices`, then call `control` on each one
   that fits. Never ask which device is meant — there is no missing device.
9. Different devices may need different actions in one request: cooling one room
   while switching another off is normal. Do not apply one action to all.
10. Never reply with your reasoning. Say what you did, or call a tool.
11. You know only this house. You cannot see the weather, the news, traffic or
   the internet. When a question needs something from outside — "is it a good
   day to do laundry", "will it rain tomorrow" — say plainly that you cannot
   see it, then answer the part you can ("the washer is free, I can start it").
   Never guess a forecast, and never answer a different question instead.
12. A house that is ALREADY in the state the request asks for is a finished
   request, not a failed one. Read the device, find the AC already cooling at
   24 °C, and the work is done: call `finish` and say so ("The bedroom AC is
   already cooling at 24 °C."). Do not change a device just to have something
   to report, and do not stop without calling `finish` — a turn that ends in
   prose is thrown away, and the user gets "I did not understand" instead."""

OFFER_SURE = """

RULE ENGINE'S READING OF THIS MESSAGE — a plan, not an order:
  intent: {intent}
  would do: {would}

The parser is confident here: it matched a phrase it knows, and the sentence
has one reading. Adopt it with `adopt_parse` unless it is impossible — the
appliance is offline, it cannot do that action, or the sentence plainly named a
different appliance. Those are things only you can see."""

OFFER_PARTIAL = """

RULE ENGINE'S READING OF THIS MESSAGE — a suggestion, not an instruction:
  intent: {intent}
  would do: {would}

The parser is not confident: this sentence has more than one reading, or this
reading covers only part of it. Check it against the house before accepting it,
and plan with the tools if it is incomplete. You are not obliged to agree with
it, and saying nothing when you disagree is not acceptable — ask, or act."""


def available(cfg: dict) -> bool:
    return llm.available(cfg)


def _system(cfg: dict, devices: list[dict], now: datetime,
            rule: dict | None = None, rule_label: str = "",
            offer: dict | None = None) -> str:
    lines = []
    for d in devices:
        lines.append(
            f"- id={d['id']} name={d['name']} type={d.get('type_label') or d['type']} "
            f"room={d.get('room') or 'unknown'} "
            f"online={'yes' if d.get('online', True) else 'no'}"
        )
    prompt = SYSTEM_PROMPT.format(
        # The scheduler uses `now`; the prompt used to use datetime.now(). On a
        # fixed clock those disagree, and the model plans against the wrong hour.
        now=now.strftime("%Y-%m-%d %H:%M:%S (%A)"),
        default_room=cfg.get("agent", {}).get("default_room", "Living Room"),
        devices="\n".join(lines),
    )
    if offer:
        # How much the parser trusts itself is information only the caller has,
        # and a model cannot infer it from the sentence — "the kitchen smells"
        # and "cool the bedroom down and shut the living room off" look equally
        # certain, and one of them is a complete plan while the other is half of
        # one. Measured: given the same permissive wording for both, the model
        # overrode the complete plan with a worse appliance in three runs out of
        # three, and had to override the half-plan to get it right.
        block = OFFER_SURE if offer.get("sure") else OFFER_PARTIAL
        prompt += block.format(intent=offer.get("intent", "?"),
                               would=offer.get("would", "?"))
    if rule:
        # Stated as a fact rather than left to the model's reading of the
        # sentence: the parser has already decided, and this is the same
        # division of labour as `schedule` refusing a model-supplied timestamp.
        nxt = nlu.compute_next(rule, now)
        when = nxt.strftime("%Y-%m-%d %H:%M") if nxt else "a later time"
        prompt += (f"\n\nTHIS REQUEST HAS A TIME: {rule_label!r}, already read as "
                   f"{when}. It happens then. Use `schedule`, never `control`.")
    return prompt


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def run(cfg: dict, text: str, adapter, now: datetime,
        default_room: str = "", prior: list[dict] | None = None,
        offer: dict | None = None, adopt=None) -> dict:
    """Run the loop. Returns {"handled": False} if the turn should fall through.

    `prior` is the transcript of a previous loop turn that ended in a question;
    the new message is appended to it so the model sees what it asked.

    `offer` + `adopt` are agent mode: the rule engine's reading of this message,
    and a callable that carries it out. Shown in the prompt, honoured only if
    the model chooses to call `adopt_parse`.
    """
    if not llm.available(cfg):
        return {"handled": False}

    devices = adapter.list_devices()
    # Parsed once, deterministically, before the model is asked anything. The
    # loop's job is to work out *what* to do; *when* was never its call.
    rule, rule_label = nlu.build_rule(text, now)
    box = ToolBox(adapter, cfg, now, default_room, rule=rule, rule_label=rule_label,
                  adopt=adopt)
    box.offer = offer
    if prior:
        messages: list[dict] = list(prior) + [{"role": "user", "content": text}]
    else:
        messages = [
            {"role": "system",
             "content": _system(cfg, devices, now, rule, rule_label, offer)},
            {"role": "user", "content": text},
        ]
    deadline = time.monotonic() + DEADLINE_SECONDS
    steps = 0
    # One nudge per stretch of prose, not one per turn. A stretch ends when the
    # model does something: the nudge is owed again once there is new
    # information to act on. Measured — "the bedroom is unbearable" was nudged
    # at step 1, went on to call list_devices and get_state, then answered in
    # prose at step 4 with the state it had just read, and the budget was
    # already spent: the turn fell to the rules' generic help. A model that only
    # talks still gets exactly one nudge, because talking never resets it.
    nudged = False

    # `adopt_parse` is only a real option when there is something to adopt. A
    # tool that answers "there is no parser reading for this turn" is a trap: a
    # model that reaches for it has understood the sentence and gets nothing
    # back, and "the bedroom is a sauna" then fell all the way through to the
    # rules' generic help.
    tools = TOOL_SCHEMAS if adopt else [
        t for t in TOOL_SCHEMAS if t["function"]["name"] != "adopt_parse"]

    try:
        while steps < MAX_STEPS and time.monotonic() < deadline:
            steps += 1
            # temperature 0: the same sentence reached the loop twice and took
            # different paths, and a demo does not survive that.
            msg = llm.chat(cfg, messages, tools=tools, temperature=0,
                           max_tokens=800)
            calls = msg.get("tool_calls") or []

            if not calls:
                reply = clean_reply(msg.get("content") or "")
                # Reporting is not doing. Asked to "wake the house up", the model
                # called list_devices, then wrote "The house has been woken up.
                # The Living Room AC, Bedroom AC..." and stopped — a claim with
                # nothing behind it, which is worse than no answer at all. One
                # nudge first, in case the model is about to act.
                if reply and not nudged and not box.targets and not box.question:
                    nudged = True
                    messages.append({"role": "assistant", "content": reply})
                    # In agent mode the nudge names the offer and forbids prose.
                    # Measured without either, "the kitchen smells" and "keep
                    # the bedroom cool every night at 11 pm" both produced a
                    # sentence instead of an action, were nudged, and produced
                    # *another* sentence — the second one literally explaining
                    # that the rule engine's reading was correct instead of
                    # accepting it, then falling back to the rules six seconds
                    # later. A model that talks about the reading has understood
                    # the sentence and still not done anything, so the nudge has
                    # to remove the option of replying in prose.
                    hint = ""
                    if box.offer:
                        hint = (f"The rule engine's reading is: {box.offer.get('would', '')}. "
                                "Call `adopt_parse` to accept it, or call `control` / "
                                "`schedule` to do it differently. ")
                    # Rule 12, said again where it is needed: the model has just
                    # described the work instead of doing it, which is the moment
                    # it decides there is nothing to do. "Already cooling at 24"
                    # is an answer; leaving it unwritten is a turn thrown away.
                    hint += ("If the house is already in the state asked for, that "
                             "is the work — call `finish` and say so. ")
                    messages.append({"role": "user", "content":
                                     "Nothing has been changed yet. Your last message described "
                                     "the work; that is not the work. " + hint +
                                     "Reply with a tool call only — no explanation. If the user "
                                     "only asked a question, call `finish` with the answer."})
                    continue
                if not reply:
                    break
                if box.question:
                    return _result(box, box.question, steps, messages, awaiting=True)
                if box.targets:
                    return _result(box, reply, steps, messages)
                # Still nothing done, and no question asked. Admit it.
                db.log("agent.loop", f"no action, no question, gave up: {reply!r}")
                return {"handled": False, "steps": steps}

            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": calls,
            })

            for call in calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                args = _parse_args(fn.get("arguments"))
                observation = box.call(name, args)
                db.log("agent.tool", f"step {steps} {name}({args}) -> "
                                     f"{'ok' if observation.get('ok') or 'devices' in observation else observation.get('error', '')}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or f"call_{steps}",
                    "content": json.dumps(observation, ensure_ascii=False),
                })
            # The model has done something, so a sentence that follows is a
            # report on new work and earns a nudge of its own.
            nudged = False

            if box.finished:
                break

        if box.question:
            return _result(box, box.question, steps, messages, awaiting=True)
        if box.reply:
            return _result(box, box.reply, steps, messages)
        if box.targets:
            return _result(box, box.summary(), steps, messages)

        # Ran out of steps or time with nothing done: say so instead of
        # pretending, and let the caller keep the "did not catch that" reply.
        return {"handled": False, "steps": steps}
    except Exception as exc:
        db.log("agent.loop", f"{type(exc).__name__}: {exc}", "error")
        return {"handled": False, "steps": steps}


def _result(box: ToolBox, reply: str, steps: int, messages: list[dict],
            awaiting: bool = False) -> dict:
    trace = list(box.trace)
    # A declined offer is a decision, and the only place it can be recorded is
    # here: the model said nothing about it, it simply did something else. Put
    # it first, because that is when it happened.
    if box.offer and trace and not box.adopted:
        trace.insert(0, {"tool": "offer", "args": {},
                         "result": f"the rules' reading: {box.offer.get('would', '')}"})
    return {
        "handled": True,
        "reply": reply,
        "ok": bool(box.targets),
        "actions": box.actions,
        "targets": box.targets,
        "steps": steps,
        "awaiting": awaiting,
        "messages": messages if awaiting else None,
        # What the model decided to do, step by step. Recorded so the interface
        # can show the work rather than assert that work happened.
        "trace": trace,
        "adopted": box.adopted,
    }
