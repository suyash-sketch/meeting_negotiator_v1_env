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
    # Priya blocked 09:00-16:00. Preferred: 16:00-17:00 GMT.
    # Jordan preferred: 11:00-12:00 EST = 16:00-17:00 GMT
    # Alex preferred: 09:00-10:00 PST = 17:00-18:00 GMT
    # Best 3-party slot: 16:00 GMT
    ("MEDIUM", 0.70, [
        Step("CheckAvailability", "REQ-MED-1", "2026-01-15T16:00Z"),
        Step("ScheduleNew", "REQ-MED-1", "2026-01-15T16:00Z"),
        Step("ScheduleNew", "REQ-MED-2", "2026-01-15T17:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── MEDIUM_B: The Blocker Sandwich ─────────────────────────────────
    # Alice (GMT) and Bob (GMT) work from 09:00Z. Dev (IST) works from 08:30Z.
    # Both blockers start at 14:00Z (Alice high) and 16:00Z (Bob medium).
    # Pre-block window: 09:00-14:00Z. Optimal slot: 13:00Z (1h, all three free,
    # and 13:00-14:00Z is within Alice's preferred 13:00-15:00 GMT).
    ("MEDIUM_B", 0.75, [
        Step("ListConflicts", "REQ-MEDB-1", "2026-01-15T13:00Z"),
        Step("ScheduleNew", "REQ-MEDB-1", "2026-01-15T13:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── MEDIUM_C: The Bump Chain ────────────────────────────────────────
    # Urgent Review (Alice+Bob) must finish by 15:00Z → schedule at 14:00Z (bumps Bob low)
    # Bumped REQ-BOB-LOW must fit before 17:00Z → schedule at 15:00Z
    ("MEDIUM_C", 0.78, [
        Step("ScheduleNew", "REQ-MEDC-URG", "2026-01-15T14:00Z"),  # bumps EVT-BOB-LOW
        Step("ScheduleNew", "REQ-BOB-LOW", "2026-01-15T15:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── HARD: The Zero-Sum Domino Cascade ──────────────────────────────
    # Must inspect preferreds first (hidden on HARD).
    # Step-through: shift Alice-Dev to 17:00, shift Alice-Bob to 16:00, schedule All-Hands at 14:00
    # Re-slot Dev low, schedule CTO sync late.
    ("HARD", 0.50, [  # unavoidable preference penalties; 0.50 = minimum viable
        Step("InspectParticipant", "CEO"),
        Step("InspectParticipant", "Alice"),
        Step("RescheduleExisting", "EVT-ALICE-DEV-URGENT", "2026-01-15T17:00Z"),
        Step("RescheduleExisting", "EVT-ALICE-BOB-URGENT", "2026-01-15T16:00Z"),
        Step("ScheduleNew", "REQ-URGENT-ALL-HANDS", "2026-01-15T14:00Z"),
        Step("ScheduleNew", "REQ-BUMPED-DEV", "2026-01-15T09:00Z"),
        Step("ScheduleNew", "REQ-CTO-SYNC", "2026-01-15T21:00Z"),
        Step("SubmitFinalCalendar"),
    ]),

    # ── HARD_B: The VIP No-Bump Gridlock ───────────────────────────────
    # CEO VIP: 14:00-17:00Z (urgent, unbumpable). Alice train: 16:00-18:00Z (high).
    # Board Prep (CEO+Alice+Bob): must go before 14:00 or after 17:00+CEO work ends (17:00 EST = 22:00Z).
    # Sprint Close (Alice+Bob): must go before Alice train at 16:00.
    ("HARD_B", 0.55, [
        Step("InspectParticipant", "CEO"),
        Step("InspectParticipant", "Alice"),
        Step("ScheduleNew", "REQ-HARDB-2", "2026-01-15T14:00Z"),   # Sprint Close: Alice+Bob before train
        Step("ScheduleNew", "REQ-HARDB-1", "2026-01-15T09:00Z"),   # Board Prep: before VIP
        Step("SubmitFinalCalendar"),
    ]),

    # ── HARD_C: The Decoy Trap ──────────────────────────────────────────
    # Trap: scheduling REQ-TRAP at 15:00 (bumps decoy) blocks REQ-REAL (needs Dev, who blocks 14-16Z).
    # Correct: schedule REQ-REAL first after Dev's blocker ends at 16:00Z, then REQ-TRAP elsewhere.
    ("HARD_C", 0.55, [
        Step("InspectParticipant", "Dev"),
        Step("ListConflicts", "REQ-HARDC-REAL", "2026-01-15T16:00Z"),
        Step("ScheduleNew", "REQ-HARDC-REAL", "2026-01-15T16:00Z"),
        Step("ScheduleNew", "REQ-HARDC-TRAP", "2026-01-15T12:00Z"),
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
