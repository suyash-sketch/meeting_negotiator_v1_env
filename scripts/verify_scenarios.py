#!/usr/bin/env python3
"""
verify_scenarios.py — Scenario solvability audit for Meeting Negotiator V1.

For each of the 9 scenarios, runs an oracle agent that executes the known-optimal
solution path and reports the terminal score. This verifies:
  1. Every scenario is solvable (score > minimum floor).
  2. Reward bounds are within spec ([0.01, 0.99]).
  3. New scenarios don't regress existing optimal scores.

Usage:
    python scripts/verify_scenarios.py              # all scenarios
    python scripts/verify_scenarios.py --tier HARD  # single tier
    python scripts/verify_scenarios.py --id EASY_C  # single scenario
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Ensure project root is importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from server.meeting_negotiator_v1_environment import MeetingNegotiatorV1Environment
from models import MeetingNegotiatorV1Action


# ── Colour helpers ─────────────────────────────────────────────────────

def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

RED    = lambda t: _c("31", t)
GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
BOLD   = lambda t: _c("1",  t)
CYAN   = lambda t: _c("36", t)


# ── Oracle action sequences (known-optimal paths) ──────────────────────

@dataclass
class Step:
    command: str
    target_id: Optional[str] = None
    proposed_start_utc: Optional[str] = None


# Each entry: (scenario_id, min_expected_score, [Step, ...])
ORACLE_PLANS: List[Tuple[str, float, List[Step]]] = [
    # ── EASY: The Empty Slate ──────────────────────────────────────────
    ("EASY", 0.85, [
        Step("ScheduleNew", "REQ-EASY-1", "2026-01-15T10:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── EASY_B: The Timezone Overlap ───────────────────────────────────
    # EST 14:00 = UTC 19:00, PST 09:00 = UTC 17:00 → overlap: 17:00-22:00 UTC
    # Both work 09:00-17:00 local → intersection: 14:00-17:00 EST = 17:00-20:00 UTC
    ("EASY_B", 0.75, [
        Step("CheckAvailability", "REQ-EASY-B1", "2026-01-15T14:00Z"),
        Step("ScheduleNew", "REQ-EASY-B1", "2026-01-15T17:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── EASY_C: The Lunch Break Gap ─────────────────────────────────────
    # Dev (IST) works 14:00-23:00 IST = 08:30-17:30 UTC
    # Alice+Bob work 09:00-12:00, 13:00-17:00 UTC
    # Combined window: 09:00-12:00 UTC
    ("EASY_C", 0.75, [
        Step("InspectParticipant", "Dev"),
        Step("ScheduleNew", "REQ-EASY-C1", "2026-01-15T09:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── MEDIUM: The Greedy Preference Trap ─────────────────────────────
    # Two-request version. The greedy move is to schedule the 3-party sync at
    # 16:00Z, which burns Priya/Jordan's best joint slot. The best verified path
    # is to place the handoff first at 16:00Z, then the 3-party sync at 17:00Z.
    ("MEDIUM", 0.94, [
        Step("ScheduleNew", "REQ-MED-2", "2026-01-15T16:00Z"),
        Step("ScheduleNew", "REQ-MED-1", "2026-01-15T17:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── MEDIUM_B: The Blocker Sandwich ─────────────────────────────────
    # Day 1 has no valid full overlap: Alice is blocked 14:00-16:00Z, Bob at
    # 16:00-17:00Z, Dev at 13:00-14:00Z. The actual solution is day 2 at 14:00Z,
    # where a low-priority placeholder for Alice+Bob can be displaced and then
    # recovered at 15:00Z for the best medium-tier score.
    ("MEDIUM_B", 0.75, [
        Step("ListConflicts", "REQ-MEDB-1", "2026-01-16T14:00Z"),
        Step("ScheduleNew", "REQ-MEDB-1", "2026-01-16T14:00Z"),
        Step("ScheduleNew", "REQ-MEDB-HOLD", "2026-01-16T15:00Z"),
        Step("ScheduleNew", "REQ-MEDB-RECOVERY", "2026-01-16T17:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── MEDIUM_C: The Bump Chain ────────────────────────────────────────
    # Urgent Review (Alice+Bob) must finish by 15:00Z → schedule at 14:00Z
    # (bumps Bob low). Bob is blocked for the rest of day 1 and from 10:00Z on
    # day 2, so the bumped low-priority event has exactly one clean recovery slot:
    # 09:00Z on day 2.
    ("MEDIUM_C", 0.78, [
        Step("ScheduleNew", "REQ-MEDC-URG", "2026-01-15T14:00Z"),  # bumps EVT-BOB-LOW
        Step("ScheduleNew", "REQ-MEDC-RECOVERY", "2026-01-15T15:00Z"),
        Step("ScheduleNew", "REQ-BOB-LOW", "2026-01-16T09:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── HARD: The Zero-Sum Domino Cascade ──────────────────────────────
    # Current best-known path inspects key participants, performs the Alice/Dev
    # and Alice/Bob cascade, schedules the urgent all-hands at 14:00Z, absorbs
    # the injected emergency debrief, re-slots the bumped Dev work, then places
    # the CTO sync late.
    ("HARD", 0.80, [
        Step("InspectParticipant", "CEO"),
        Step("InspectParticipant", "Alice"),
        Step("RescheduleExisting", "EVT-ALICE-DEV-HIGH", "2026-01-15T17:00Z"),
        Step("RescheduleExisting", "EVT-ALICE-BOB-HIGH", "2026-01-15T16:00Z"),
        Step("ScheduleNew", "REQ-URGENT-ALL-HANDS", "2026-01-15T14:00Z"),
        Step("ScheduleNew", "REQ-HARD-FU1", "2026-01-15T18:00Z"),
        Step("ScheduleNew", "REQ-BUMPED-DEV", "2026-01-15T09:00Z"),
        Step("ScheduleNew", "REQ-CTO-SYNC", "2026-01-15T21:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── HARD_B: The 60-Minute VIP Bottleneck ───────────────────────────
    # The current version allows the urgent 3-way sync to displace the protected
    # CEO block, which injects a bumped replacement request plus an executive
    # recovery sync. The best verified path resolves both follow-ons and then
    # places the decoy on day 2.
    ("HARD_B", 0.87, [
        Step("InspectParticipant", "CEO"),
        Step("ScheduleNew", "REQ-HARDB-ALL", "2026-01-15T17:00Z"),
        Step("RescheduleExisting", "EVT-ALICE-MID", "2026-01-15T09:00Z"),
        Step("ScheduleNew", "REQ-BUMP-EVT-CEO-URGENT", "2026-01-15T16:00Z"),
        Step("InspectParticipant", "CTO"),
        Step("InspectParticipant", "CEO"),
        Step("InspectParticipant", "CTO"),
        Step("ScheduleNew", "REQ-HARDB-EXEC-RECOVERY", "2026-01-15T18:00Z"),
        Step("InspectParticipant", "CEO"),
        Step("InspectParticipant", "CTO"),
        Step("ScheduleNew", "REQ-HARDB-DECOY", "2026-01-16T17:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── HARD_C: The 48-Hour Blind Cascade ───────────────────────────────
    # Current hard scenario has one urgent three-way meeting on day 1. It only
    # fits at 15:00Z by displacing the protected trap block, which then creates
    # an explicit board-recovery sync that should be cleared before submission.
    ("HARD_C", 0.55, [
        Step("InspectParticipant", "CEO"),
        Step("ScheduleNew", "REQ-URGENT-DAY1", "2026-01-15T15:00Z"),
        Step("ScheduleNew", "REQ-HARDC-DECISION-RECOVERY", "2026-01-15T17:00Z"),
        Step("SubmitFinalCalendar"),
    ]),
]

TIER_MAP = {
    "EASY": "easy", "EASY_B": "easy", "EASY_C": "easy",
    "MEDIUM": "medium", "MEDIUM_B": "medium", "MEDIUM_C": "medium",
    "HARD": "hard", "HARD_B": "hard", "HARD_C": "hard",
}


# ── Runner ─────────────────────────────────────────────────────────────

def run_oracle(scenario_id: str, steps: List[Step]) -> Tuple[float, Dict, int, bool]:
    """Execute the oracle plan and return (score, breakdown, turn_count, passed)."""
    env = MeetingNegotiatorV1Environment()
    obs = env.reset(scenario_id=scenario_id)
    turn = 0

    for step in steps:
        action = MeetingNegotiatorV1Action(
            command=step.command,
            target_id=step.target_id,
            proposed_start_utc=step.proposed_start_utc,
        )
        obs = env.step(action)
        turn += 1
        if obs.done:
            break

    score = obs.score or 0.01
    breakdown = obs.reward_breakdown or {}
    return score, breakdown, turn, obs.done


def fmt_breakdown(bd: Dict) -> str:
    parts = []
    for k, v in bd.items():
        label = k.replace("_", " ")
        colour = GREEN if v > 0.05 else (YELLOW if v > 0 else RED)
        parts.append(f"{label}={colour(f'{v:.3f}')}")
    return "  " + " | ".join(parts)


def run_suite(filter_id: Optional[str] = None, filter_tier: Optional[str] = None):
    plans = ORACLE_PLANS
    if filter_id:
        plans = [(sid, mn, steps) for sid, mn, steps in plans if sid == filter_id.upper()]
    if filter_tier:
        plans = [(sid, mn, steps) for sid, mn, steps in plans
                 if TIER_MAP.get(sid, "") == filter_tier.lower()]

    if not plans:
        print(RED(f"No matching scenarios (id={filter_id}, tier={filter_tier})."))
        sys.exit(1)

    print(BOLD(f"\n{'─'*58}"))
    print(BOLD(f"  Meeting Negotiator V1 — Scenario Solvability Audit"))
    print(BOLD(f"{'─'*58}"))

    results = []
    for scenario_id, min_score, steps in plans:
        tier = TIER_MAP.get(scenario_id, "?")
        score, breakdown, turns, done = run_oracle(scenario_id, steps)
        ok = score >= min_score and done and 0.01 <= score <= 0.99

        status = GREEN("PASS") if ok else RED("FAIL")
        score_str = GREEN(f"{score:.4f}") if score >= min_score else RED(f"{score:.4f}")
        print(f"\n[{status}] {BOLD(scenario_id)} ({tier}) | score={score_str} "
              f"(min={min_score}) | turns={turns} | done={done}")
        if breakdown:
            print(fmt_breakdown(breakdown))
        if not ok:
            reasons = []
            if score < min_score:
                reasons.append(f"score {score:.4f} < min {min_score}")
            if not done:
                reasons.append("episode did not reach done=True")
            if not (0.01 <= score <= 0.99):
                reasons.append(f"score {score:.4f} outside [0.01, 0.99]")
            print(f"  {RED('Reason:')} {'; '.join(reasons)}")

        results.append(ok)

    passed = sum(results)
    total = len(results)
    print(BOLD(f"\n{'─'*58}"))
    if passed == total:
        print(GREEN(BOLD(f"  All {passed}/{total} scenarios passed.")))
    else:
        print(RED(BOLD(f"  {passed}/{total} scenarios passed. {total-passed} FAILED.")))
    print(BOLD(f"{'─'*58}\n"))
    return passed == total


def main():
    parser = argparse.ArgumentParser(
        description="Verify all 9 Meeting Negotiator scenarios are solvable."
    )
    parser.add_argument("--id", help="Run a single scenario by ID (e.g. HARD_C)")
    parser.add_argument("--tier", help="Run all scenarios for a tier (easy/medium/hard)")
    args = parser.parse_args()
    ok = run_suite(filter_id=args.id, filter_tier=args.tier)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
