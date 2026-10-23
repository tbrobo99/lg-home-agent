"""Risk-graded action policy (Phase 2).

The proposal's central safety claim: actions are split into two tiers, and no
Tier-2 action ever executes without explicit user approval. This module is
where that claim is enforced — the only place in the codebase that decides
whether a proposed action may run on its own.

Tier 1 (automatic, logged, reported): drum micro-ventilation, dehumidifier,
air purifier, kitchen exhaust. Reversible, low-power, and — the property that
actually matters — worst case is a slightly dry room or a fan running for
nothing, never a ruined garment or a delayed meal.

Tier 2 (approval required): anything that moves a *schedule* or touches a
*heating* appliance. Delaying the dryer and restarting it off-peak is the
realised case: it changes what the user planned, so the user decides.

`candidates()` turns a risk into the actions worth considering. It deliberately
produces a *set*, not a plan: the decision layer chooses among these, in the
order it defends. Rules propose; the agent disposes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TIER1 = "tier1"
TIER2 = "tier2"

# The whitelist. An action kind that is not in this table does not exist as far
# as the policy layer is concerned — nothing downstream can invent one.
#
# The table is exactly what `candidates()` can produce, and a test enforces
# that: an entry no rule ever proposes is a capability the code claims and does
# not have. Two earlier entries were removed for that reason —
# `reschedule_washer` and `preheat_cooktop`. The proposal names "delaying a
# laundry cycle or warming up cooktops" as *examples* of Tier-2 work, not as
# required actions; the realised schedule shift is `delay_dryer`, and there is
# no cooktop in the appliance set to warm. Adding either back is a feature, and
# it starts here.
ACTION_TIERS: dict[str, str] = {
    "ventilate_washer": TIER1,
    "dehumidifier_on": TIER1,
    "purifier_on": TIER1,
    "ventilate_kitchen": TIER1,
    "delay_dryer": TIER2,
}

# How long a Tier-2 request waits before the fail-safe timeout (poster §II.E).
APPROVAL_TTL_SECONDS = 300


@dataclass
class Candidate:
    kind: str
    device_id: str
    device_name: str
    actions: list[dict]
    label: str            # one line, shown to the user and the LLM
    tier: str = ""
    # Past-tense form for the report the user reads *after* the action ran:
    # "I started drum micro-ventilation", not "I start drum micro-ventilation".
    done: str = ""
    # For Tier-2 schedule shifts: the job to create if the user confirms.
    job: dict | None = None
    reason: str = ""      # why the rule proposed it — evidence, not decoration

    def __post_init__(self) -> None:
        if not self.tier:
            self.tier = tier_of(self.kind)
        if not self.done:
            self.done = self.label


def tier_of(kind: str) -> str:
    return ACTION_TIERS.get(kind, TIER2)  # unknown kinds default to the safe tier


def candidates(risk, snapshot: dict) -> list[Candidate]:
    """Propose actions for a fired risk. Order is a suggestion, not a decision."""
    ap = snapshot["appliances"]
    rooms = snapshot["rooms"]
    out: list[Candidate] = []

    if risk.kind == "mold":
        w = ap["washer"]
        if not w.get("ventilating"):
            out.append(Candidate(
                kind="ventilate_washer",
                device_id="washer",
                device_name="Balcony Washer",
                actions=[{"action": "ventilate", "value": True}],
                label="start drum micro-ventilation on the washer",
                done="started drum micro-ventilation on the washer",
                reason=f"drum at {risk.current:.0f}% humidity, sealed {risk.drivers.get('sealed_hours', 0):.1f} h",
            ))
        balcony = rooms.get("Balcony", {}).get("humidity", 60)
        if balcony > 55 and not ap["dehumidifier"].get("power"):
            out.append(Candidate(
                kind="dehumidifier_on",
                device_id="dehumidifier",
                device_name="Balcony Dehumidifier",
                actions=[{"action": "power", "value": True}],
                label="turn on the balcony dehumidifier",
                done="turned on the balcony dehumidifier",
                reason=f"balcony humidity {balcony:.0f}%",
            ))
        # A damp drum is also an airborne problem: the flat breathes what the
        # balcony grows. The purifier is the cheapest reversible answer, and
        # the proposal names it explicitly as a Tier-1 action.
        if not ap.get("purifier", {}).get("power"):
            out.append(Candidate(
                kind="purifier_on",
                device_id="purifier",
                device_name="Living Room Purifier",
                actions=[{"action": "power", "value": True}],
                label="turn on the living room air purifier to catch mould spores",
                done="turned on the living room air purifier",
                reason=(f"mould risk {risk.score:.0f}/100; a wet drum that has been "
                        f"sealed for hours is shedding spores into the flat"),
            ))

    elif risk.kind == "thermal":
        if ap["dryer"].get("power"):
            out.append(Candidate(
                kind="delay_dryer",
                device_id="dryer",
                device_name="Balcony Dryer",
                actions=[{"action": "power", "value": False}],
                label="turn the dryer off now and restart it at 11 PM tonight",
                done="turned the dryer off and set it to restart at 11 PM",
                tier=TIER2,
                job={"device_id": "dryer",
                     "actions": [{"action": "power", "value": True}],
                     "rule": {"type": "once", "run_at": _next_2300()},
                     "repeat_label": "one time at 23:00"},
                reason=f"{snapshot.get('heat_load_w', 0)} W of heat load, kitchen at {risk.current:.1f}\u00b0C",
            ))
        # The kitchen has no AC, so exhausting it is the whole move. The fridge
        # duty is the evidence: past 60% the compressor is paying for the heat.
        hood = ap.get("range_hood", {})
        if not hood.get("power"):
            out.append(Candidate(
                kind="ventilate_kitchen",
                device_id="range_hood",
                device_name="Kitchen Range Hood",
                actions=[{"action": "power", "value": True},
                         {"action": "fan", "value": "high"}],
                label="turn on the kitchen range hood to exhaust the heat",
                done="turned on the kitchen range hood to pull the heat out",
                reason=(f"fridge compressor at "
                        f"{snapshot['appliances'].get('fridge', {}).get('duty_cycle', 0):.0%} duty, "
                        f"predicted {risk.predicted:.1f}\u00b0C in "
                        f"{risk.drivers.get('oven_scheduled_in_min') or 30} min"),
            ))

    return out


def _next_2300(now=None) -> str:
    from datetime import datetime, timedelta
    now = now or datetime.now()
    target = now.replace(hour=23, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target.isoformat(timespec="seconds")
