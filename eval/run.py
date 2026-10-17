"""Reproducible evaluation of the intent parser.

    python -m eval.run              # summary + failures
    python -m eval.run --verbose    # per-case results
    python -m eval.run --min 1.0    # fail unless everything passes

No network, no API key, no LLM. The point is that the claim "it understands the
common cases" is a number you can re-run, not a feeling you get from a demo.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import nlu  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
NOW = datetime(2026, 10, 17, 18, 42)  # Saturday evening, after noon

DEVICES = [
    {"id": "ac_living", "name": "Living Room AC", "type": "air_conditioner",
     "room": "Living Room", "online": True},
    {"id": "ac_bedroom", "name": "Bedroom AC", "type": "air_conditioner",
     "room": "Bedroom", "online": True},
    {"id": "purifier", "name": "Living Room Purifier", "type": "air_purifier",
     "room": "Living Room", "online": True},
    {"id": "washer", "name": "Balcony Washer", "type": "washer",
     "room": "Balcony", "online": True},
    {"id": "fridge", "name": "Kitchen Fridge", "type": "refrigerator",
     "room": "Kitchen", "online": True},
]


def load_cases() -> list[dict]:
    cases = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def action_set(actions: list[dict]) -> set:
    out = set()
    for a in actions or []:
        v = a.get("value")
        if isinstance(v, bool) or v is None:
            out.add((a.get("action"), v))
        else:
            out.add((a.get("action"), v))
    return out


def expected_actions(case: dict) -> set | None:
    if "actions" not in case:
        return None
    return {(name, value) for name, value in case["actions"]}


def check(case: dict) -> tuple[bool, str]:
    got = nlu.parse(case["text"], DEVICES, NOW, "Living Room")

    if got["intent"] != case["intent"]:
        return False, f'intent: expected {case["intent"]}, got {got["intent"]}'

    exp_actions = expected_actions(case)
    if exp_actions is not None and action_set(got["actions"]) != exp_actions:
        return False, f'actions: expected {sorted(exp_actions)}, got {sorted(action_set(got["actions"]))}'

    if "device" in case and got.get("device_id") != case["device"]:
        return False, f'device: expected {case["device"]}, got {got.get("device_id")}'

    if "targets" in case:
        got_ids = sorted(t["id"] for t in got.get("targets") or [])
        if got_ids != sorted(case["targets"]):
            return False, f'targets: expected {sorted(case["targets"])}, got {got_ids}'

    if "rule" in case:
        if got.get("rule") != case["rule"]:
            return False, f'rule: expected {case["rule"]}, got {got.get("rule")}'

    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--min", type=float, default=0.95,
                    help="minimum pass rate (default 0.95)")
    args = ap.parse_args()

    cases = load_cases()
    results = []
    for case in cases:
        ok, why = check(case)
        results.append((case, ok, why))

    by_cat: dict[str, list[int]] = {}
    for case, ok, _ in results:
        row = by_cat.setdefault(case["category"], [0, 0])
        row[1] += 1
        row[0] += 1 if ok else 0

    print(f"Evaluating {len(cases)} utterances against the rule engine "
          f"(no network, no LLM, fixed clock 2026-10-17 18:42)\n")
    print(f"{'category':<16}{'cases':>7}{'pass':>7}{'acc':>8}")
    print("-" * 38)
    for cat in sorted(by_cat):
        ok, total = by_cat[cat]
        print(f"{cat:<16}{total:>7}{ok:>7}{ok / total:>7.0%}")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    rate = passed / total
    print("-" * 38)
    print(f"{'overall':<16}{total:>7}{passed:>7}{rate:>7.1%}\n")

    failures = [(c, why) for c, ok, why in results if not ok]
    if failures:
        print(f"{len(failures)} failure(s):")
        for case, why in failures:
            print(f'  [{case["category"]}] "{case["text"]}"')
            print(f"      {why}")
        print()
    elif args.verbose:
        for case, _, _ in results:
            print(f'  ok  [{case["category"]}] "{case["text"]}"')

    if rate < args.min:
        print(f"FAIL: {rate:.1%} is below the {args.min:.0%} threshold")
        return 1
    print(f"PASS: {rate:.1%} >= {args.min:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
