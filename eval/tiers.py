"""Which tier answers which sentence, and what it costs.

    python -m eval.tiers                 # table + summary
    python -m eval.tiers --markdown      # the table, for the README
    python -m eval.tiers --only loop     # just the sentences tier 3 handled

Needs a working `llm` block in config.json, because the question it answers is
about the cascade, not about the rules. `eval.run` is the offline one.

The sentences are deliberately the ones the rule engine cannot parse. A cascade
is only worth having if the tiers above the rules actually get used, and that is
easy to lose: hand a request to a blind single pass that answers everything and
the tool loop becomes dead code that still looks installed.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, db, llm  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.devices import MockAdapter  # noqa: E402
from app.devices import mock as mock_mod  # noqa: E402

SENTENCES = [
    "the bedroom is a sauna, sort it out",
    "the whole flat is too warm",
    "put the whole flat to sleep",
    "put the living room to sleep in 20 minutes",
    "the air in here is disgusting",
    "wake the house up",
    "get the flat ready for bed",
    "the fridge is too warm",
    "everything off, I'm leaving",
    "it feels clammy in the bedroom",
    "I'm freezing in the bedroom",
    "the kitchen smells",
    "the washer has been on for ages",
    "cool the bedroom down and shut the living room off",
    "keep the bedroom cool every night at 11 pm",
]

# `agent.handle` reports which tier answered, because intent names do not: a
# tier 2 plan and a rule match both come back as "control".
TIER = {"rules": "1 rules", "scene": "1 scene", "llm": "2 single-shot",
        "loop": "3 loop", "agent": "3 agent"}


def _actions(actions: list[dict]) -> str:
    return ", ".join(f"{a.get('action')}={a.get('value')}" for a in actions or [])


def _one(cfg: dict, text: str) -> dict:
    """One sentence, one fresh agent — so no run can prime the next."""
    adapter = MockAdapter()
    try:
        agent = Agent(adapter, cfg)
        started = time.monotonic()
        try:
            result = agent.handle(text)
        except Exception as exc:                      # noqa: BLE001
            result = {"intent_name": "error", "reply": f"{type(exc).__name__}: {exc}",
                      "ok": False, "actions": []}
        elapsed = int((time.monotonic() - started) * 1000)
        name = result.get("intent_name") or "unknown"
        tier = result.get("tier") or "rules"
        return {
            "text": text,
            "intent": name,
            "tier": TIER.get(tier, tier),
            "steps": result.get("steps"),
            "asked": bool(result.get("awaiting")),
            "ms": elapsed,
            "ok": bool(result.get("ok")),
            "actions": _actions(result.get("actions")),
            "reply": (result.get("reply") or "").replace("\n", " / "),
        }
    finally:
        adapter.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true", help="emit a markdown table")
    ap.add_argument("--only", choices=["1", "2", "3"], help="filter by tier")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    ap.add_argument("--agent", action="store_true",
                    help="agent mode: the rules offer their reading to the "
                         "model instead of applying it")
    args = ap.parse_args()

    cfg = config.load()
    if not llm.available(cfg):
        print("No llm.api_key in config.json — the cascade needs a model.")
        return 2
    # Agent mode is measured on the same sentences as the cascade. That is the
    # only comparison that says anything: what does letting the model decide
    # buy, and what does it cost in time.
    cfg.setdefault("agent", {})["mode"] = "agent" if args.agent else "cascade"

    tmp = tempfile.TemporaryDirectory()
    orig = (db.DATA_DIR, db.DB_PATH, mock_mod.STATE_PATH)
    db.DATA_DIR = Path(tmp.name)
    db.DB_PATH = Path(tmp.name) / "eval.db"
    mock_mod.STATE_PATH = Path(tmp.name) / "mock_state.json"
    db.init()

    rows = []
    try:
        for text in SENTENCES:
            rows.append(_one(cfg, text))
    finally:
        db.DATA_DIR, db.DB_PATH, mock_mod.STATE_PATH = orig
        tmp.cleanup()

    if args.only:
        rows = [r for r in rows if r["tier"].startswith(args.only)]

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    model = cfg["llm"].get("model")
    if args.markdown:
        print("| utterance | tier | steps | time | what it did |")
        print("| --- | --- | --- | --- | --- |")
        for r in rows:
            steps = r["steps"] or "—"
            if r["asked"]:
                did = f'asked: {r["reply"][:60]}'
            else:
                did = r["actions"] or r["reply"][:64]
            print(f'| "{r["text"]}" | {r["tier"]} | {steps} | {r["ms"] / 1000:.1f}s | {did} |')
        return 0

    label = "Agent mode" if args.agent else "Cascade"
    print(f'{label} over {len(rows)} utterances rules cannot parse '
          f'(mock devices, model {model})\n')
    print(f'{"tier":<15}{"steps":>6}{"time":>7}  utterance')
    print("-" * 78)
    for r in rows:
        steps = r["steps"] or "-"
        print(f'{r["tier"]:<15}{steps:>6}{r["ms"] / 1000:>6.1f}s  {r["text"]}')

    # Agent mode reports "3 agent" for the same loop, so count both names —
    # otherwise the summary says "loop 0" while nine rows went through one.
    loop = [r for r in rows if r["tier"] in ("3 loop", "3 agent")]
    single = [r for r in rows if r["tier"] == "2 single-shot"]
    asked = [r for r in rows if r["asked"]]
    # "no answer" means the cascade stopped short, not that a device refused:
    # a loop that reports a refusal has still understood the request.
    no_answer = [r for r in rows
                 if r["intent"] in ("unknown", "error")
                 or r["intent"].startswith("need_")]
    total_ms = sum(r["ms"] for r in rows)

    print("-" * 78)
    print(f'loop {len(loop)}   single-shot {len(single)}   asked {len(asked)}   '
          f'no answer {len(no_answer)}   total {total_ms / 1000:.1f}s')
    if loop:
        avg = sum(r["ms"] for r in loop) / len(loop) / 1000
        steps = [r["steps"] for r in loop if r["steps"]]
        print(f'avg loop latency {avg:.1f}s over {len(steps)} runs, '
              f'{min(steps)}–{max(steps)} steps')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
