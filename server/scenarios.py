"""Scenario definitions for Meeting Negotiator V1.

All 9 scenarios across 3 difficulty tiers (3 per tier).
Scenarios are loaded by name via `get_scenario(scenario_id)`.

Tier overview:
  EASY   — single pending request, straightforward constraints
  EASY_B — cross-timezone overlap, non-overlapping preferences
  EASY_C — 3-party with IST participant + lunch break gap

  MEDIUM   — multi-request, preference trap, dynamic followup
  MEDIUM_B — blocker sandwich, urgent override
  MEDIUM_C — bump chain: urgent forces lower-priority re-slot

  HARD   — zero-sum domino cascade + dynamic emergency followup
  HARD_B — VIP no-bump gridlock, multi-day window
  HARD_C — decoy trap: low-priority event blocks critical slot
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from ..models import Participant, MeetingRequest, ScheduledEvent
except ImportError:
    import os as _os
    import sys as _sys
    _parent_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _parent_dir not in _sys.path:
        _sys.path.insert(0, _parent_dir)
    from models import Participant, MeetingRequest, ScheduledEvent


# ── Scenario Container ─────────────────────────────────────────────────

@dataclass(frozen=True)
class ScenarioSpec:
    """A fully-specified episode for the Meeting Negotiator environment."""
    scenario_id: str
    description: str
    current_time_utc: str
    participants: Dict[str, Participant]
    calendar_state: List[ScheduledEvent]
    pending_requests: List[MeetingRequest]
    all_requests: List[MeetingRequest]
    max_turns: int = 15
    dynamic_followups: Optional[List[MeetingRequest]] = None


# ── EASY Tier ──────────────────────────────────────────────────────────

def scenario_easy() -> ScenarioSpec:
    """EASY: The Empty Slate.
    A simple 1:1 meeting between two UTC participants with shared working hours.
    No calendar blockers. Optimal: 1 ScheduleNew + 1 Submit (score ≥ 0.90).
    """
    participants = {
        "Alice": Participant(
            name="Alice", timezone="UTC",
            working_hours=["09:00-12:00", "13:00-17:00"],
            preferred_hours=["10:00-11:30", "14:00-16:00"],
        ),
        "Bob": Participant(
            name="Bob", timezone="UTC",
            working_hours=["09:00-12:00", "13:00-17:00"],
            preferred_hours=["09:30-11:00", "15:00-16:30"],
        ),
    }
    request = MeetingRequest(
        request_id="REQ-EASY-1", attendees=["Alice", "Bob"],
        duration_minutes=60, priority="medium",
        deadline_utc="2026-01-15T17:00Z", title="1:1 Alice/Bob",
    )
    return ScenarioSpec(
        scenario_id="EASY", description="The Empty Slate",
        current_time_utc="2026-01-15T08:00Z",
        participants=participants,
        calendar_state=[],
        pending_requests=[request], all_requests=[request],
        max_turns=9,
    )


def scenario_easy_b() -> ScenarioSpec:
    """EASY_B: The Timezone Overlap.
    EST and PST participants with non-overlapping preferred windows.
    Agent must find a UTC slot where both working hours intersect.
    Optimal: CheckAvailability + ScheduleNew + Submit.
    """
    participants = {
        "Alice": Participant(
            name="Alice", timezone="EST",
            working_hours=["09:00-17:00"],
            preferred_hours=["09:00-10:00"],
        ),
        "Bob": Participant(
            name="Bob", timezone="PST",
            working_hours=["09:00-17:00"],
            preferred_hours=["16:00-17:00"],
        ),
    }
    request = MeetingRequest(
        request_id="REQ-EASY-B1", attendees=["Alice", "Bob"],
        duration_minutes=60, priority="medium",
        deadline_utc="2026-01-15T23:00Z", title="Cross-TZ Sync",
    )
    return ScenarioSpec(
        scenario_id="EASY_B", description="The Timezone Overlap",
        current_time_utc="2026-01-15T08:00Z",
        participants=participants,
        calendar_state=[],
        pending_requests=[request], all_requests=[request],
        max_turns=9,
    )


def scenario_easy_c() -> ScenarioSpec:
    """EASY_C: The Lunch Break Gap.
    Three-party meeting including an IST participant whose working hours
    start at 14:30 IST (09:00 UTC). Agents must account for the IST offset
    and avoid scheduling across the Alice/Bob lunch break.
    Optimal: InspectParticipant(Dev) + ScheduleNew + Submit.
    """
    participants = {
        "Alice": Participant(
            name="Alice", timezone="UTC",
            working_hours=["09:00-12:00", "13:00-17:00"],
            preferred_hours=["10:00-12:00"],
        ),
        "Bob": Participant(
            name="Bob", timezone="UTC",
            working_hours=["09:00-12:00", "13:00-17:00"],
            preferred_hours=["10:00-12:00"],
        ),
        "Dev": Participant(
            name="Dev", timezone="IST",
            working_hours=["14:00-23:00"],  # 08:30–17:30 UTC
            preferred_hours=["14:30-16:30"],  # 09:00–11:00 UTC
        ),
    }
    request = MeetingRequest(
        request_id="REQ-EASY-C1", attendees=["Alice", "Bob", "Dev"],
        duration_minutes=60, priority="high",
        deadline_utc="2026-01-15T17:00Z", title="Tri-Party Standup",
    )
    return ScenarioSpec(
        scenario_id="EASY_C", description="The Lunch Break Gap",
        current_time_utc="2026-01-15T08:00Z",
        participants=participants,
        calendar_state=[],
        pending_requests=[request], all_requests=[request],
        max_turns=9,
    )


# ── MEDIUM Tier ────────────────────────────────────────────────────────

def scenario_medium() -> ScenarioSpec:
    """MEDIUM: The Greedy Preference Trap.
    Three participants across GMT/EST/PST with a large morning blocker on Priya.
    Two pending requests + a dynamic follow-up injected mid-episode.
    Greedy agents schedule the first slot found and miss Priya's preferences.
    Optimal: CheckAvailability on preferred windows + 2x ScheduleNew + Submit.
    """
    participants = {
        "Priya": Participant(
            name="Priya", timezone="GMT",
            working_hours=["09:00-18:00"],
            preferred_hours=["16:00-17:00"],
        ),
        "Jordan": Participant(
            name="Jordan", timezone="EST",
            working_hours=["09:00-17:00"],
            preferred_hours=["11:00-12:00"],
        ),
        "Alex": Participant(
            name="Alex", timezone="PST",
            working_hours=["08:00-17:00"],
            preferred_hours=["09:00-10:00"],
        ),
    }
    calendar_state = [
        ScheduledEvent(
            event_id="EVT-PRIYA-BLOCK", attendees=["Priya"],
            start_time_utc="2026-01-15T09:00Z", duration_minutes=420, priority="medium",
        ),
    ]
    req_1 = MeetingRequest(
        request_id="REQ-MED-1", attendees=["Priya", "Jordan", "Alex"],
        duration_minutes=60, priority="high",
        deadline_utc="2026-01-15T18:00Z", title="Full Team Sync",
    )
    req_2 = MeetingRequest(
        request_id="REQ-MED-2", attendees=["Priya", "Jordan"],
        duration_minutes=60, priority="medium",
        deadline_utc="2026-01-15T18:00Z", title="Priya-Jordan Handoff",
    )
    followup = MeetingRequest(
        request_id="REQ-MED-FU1", attendees=["Priya", "Alex"],
        duration_minutes=30, priority="low",
        deadline_utc="2026-01-15T20:00Z", title="Follow-up Notes",
    )
    return ScenarioSpec(
        scenario_id="MEDIUM", description="The Greedy Preference Trap",
        current_time_utc="2026-01-15T15:00Z",
        participants=participants, calendar_state=calendar_state,
        pending_requests=[req_1, req_2],
        all_requests=[req_1, req_2, followup],
        max_turns=12, dynamic_followups=[followup],
    )


def scenario_medium_b() -> ScenarioSpec:
    """MEDIUM_B: The Blocker Sandwich.
    A multiday urgent 3-party meeting. Day 1 has tempting partial gaps, but
    no valid full overlap for all three attendees. The real solution is on day
    2, where a low-priority placeholder must be displaced and then recovered at
    its only remaining slot. Agents that only search the current day waste
    turns on impossible same-day slots.

    Optimal: ListConflicts on the day-2 slot + ScheduleNew(bump low hold) +
    ScheduleNew(recover low hold) + Submit.
    """
    participants = {
        "Alice": Participant(
            name="Alice", timezone="EST",
            working_hours=["09:00-18:00"],
            preferred_hours=["09:00-11:00"],
        ),
        "Bob": Participant(
            name="Bob", timezone="GMT",
            working_hours=["09:00-18:00"],
            preferred_hours=["15:00-17:00"],
        ),
        "Dev": Participant(
            name="Dev", timezone="IST",
            working_hours=["14:00-23:00"],  # 08:30–17:30 UTC
            preferred_hours=["18:00-20:00"],  # 12:30–14:30 UTC
        ),
    }
    calendar_state = [
        ScheduledEvent(
            event_id="EVT-ALICE-BLOCK", attendees=["Alice"],
            start_time_utc="2026-01-15T14:00Z", duration_minutes=120, priority="high",
        ),
        ScheduledEvent(
            event_id="EVT-BOB-BLOCK", attendees=["Bob"],
            start_time_utc="2026-01-15T16:00Z", duration_minutes=60, priority="medium",
        ),
        ScheduledEvent(
            event_id="EVT-DEV-DAY1-BLOCK", attendees=["Dev"],
            start_time_utc="2026-01-15T13:00Z", duration_minutes=60, priority="medium",
        ),
        ScheduledEvent(
            event_id="EVT-DAY2-HOLD", attendees=["Alice", "Bob"],
            start_time_utc="2026-01-16T14:00Z", duration_minutes=60, priority="low",
            request_id="REQ-MEDB-HOLD",
        ),
        ScheduledEvent(
            event_id="EVT-BOB-DAY2-BLOCK", attendees=["Bob"],
            start_time_utc="2026-01-16T16:00Z", duration_minutes=60, priority="medium",
        ),
    ]
    req = MeetingRequest(
        request_id="REQ-MEDB-1", attendees=["Alice", "Bob", "Dev"],
        duration_minutes=60, priority="urgent",
        deadline_utc="2026-01-16T15:00Z", title="Emergency Sync",
    )
    req_hold = MeetingRequest(
        request_id="REQ-MEDB-HOLD", attendees=["Alice", "Bob"],
        duration_minutes=60, priority="low",
        deadline_utc="2026-01-16T16:00Z", title="Placeholder Hold",
    )
    return ScenarioSpec(
        scenario_id="MEDIUM_B", description="The Blocker Sandwich",
        current_time_utc="2026-01-15T13:00Z",
        participants=participants, calendar_state=calendar_state,
        pending_requests=[req], all_requests=[req, req_hold],
        max_turns=10,
    )


def scenario_medium_c() -> ScenarioSpec:
    """MEDIUM_C: The Bump Chain.
    An urgent meeting must be scheduled at 14:00 UTC, but Bob already has a
    low-priority event there. That bump is only the first half of the task:
    the displaced event cannot be recovered later the same day because of Bob's
    afternoon blockers, so it must be moved to the unique next-morning slot.

    Optimal: ScheduleNew (bump) + ScheduleNew (recover next morning) + Submit.
    """
    participants = {
        "Alice": Participant(
            name="Alice", timezone="EST",
            working_hours=["09:00-17:00"],
            preferred_hours=["10:00-12:00"],
        ),
        "Bob": Participant(
            name="Bob", timezone="UTC",
            working_hours=["09:00-17:00"],
            preferred_hours=["14:00-16:00"],
        ),
    }
    existing = ScheduledEvent(
        event_id="EVT-BOB-LOW", attendees=["Bob"],
        start_time_utc="2026-01-15T14:00Z", duration_minutes=60, priority="low",
        request_id="REQ-BOB-LOW",
    )
    late_block = ScheduledEvent(
        event_id="EVT-BOB-LATE", attendees=["Bob"],
        start_time_utc="2026-01-15T15:00Z", duration_minutes=120, priority="medium",
    )
    next_day_block = ScheduledEvent(
        event_id="EVT-BOB-NEXTDAY", attendees=["Bob"],
        start_time_utc="2026-01-16T10:00Z", duration_minutes=420, priority="medium",
    )
    req_urgent = MeetingRequest(
        request_id="REQ-MEDC-URG", attendees=["Alice", "Bob"],
        duration_minutes=60, priority="urgent",
        deadline_utc="2026-01-15T15:00Z", title="Urgent Review",
    )
    req_low = MeetingRequest(
        request_id="REQ-BOB-LOW", attendees=["Bob"],
        duration_minutes=60, priority="low",
        deadline_utc="2026-01-16T10:00Z", title="Bob Background",
    )
    return ScenarioSpec(
        scenario_id="MEDIUM_C", description="The Bump Chain",
        current_time_utc="2026-01-15T12:00Z",
        participants=participants, calendar_state=[existing, late_block, next_day_block],
        pending_requests=[req_urgent], all_requests=[req_urgent, req_low],
        max_turns=10,
    )


# ── HARD Tier ──────────────────────────────────────────────────────────

def scenario_hard() -> ScenarioSpec:
    """HARD: The Zero-Sum Domino Cascade.
    Fully occupied calendar for most participants. Two pending requests with
    very different urgencies. Preferred hours are HIDDEN (HARD tier partial
    observability). A dynamic emergency follow-up fires mid-episode.
    Optimal: InspectParticipant(s) + cascade of bumps + emergency re-slot.
    """
    participants = {
        "CEO": Participant(
            name="CEO", timezone="EST",
            working_hours=["08:00-17:00"],
            preferred_hours=["16:00-17:00"],
        ),
        "CTO": Participant(
            name="CTO", timezone="PST",
            working_hours=["09:00-18:00"],
            preferred_hours=["10:00-11:00"],
        ),
        "Alice": Participant(
            name="Alice", timezone="EST",
            working_hours=["09:00-17:00"],
            preferred_hours=["16:00-17:00"],
        ),
        "Bob": Participant(
            name="Bob", timezone="GMT",
            working_hours=["08:00-18:00"],
            preferred_hours=["16:00-17:00"],
        ),
        "Dev": Participant(
            name="Dev", timezone="IST",
            working_hours=["14:00-23:59"],
            preferred_hours=["14:30-16:30"],
        ),
    }
    calendar_state = [
        ScheduledEvent(event_id="EVT-BOB-MORNING", attendees=["Bob"],
            start_time_utc="2026-01-15T08:00Z", duration_minutes=360, priority="urgent"),
        ScheduledEvent(event_id="EVT-BOB-MID", attendees=["Bob"],
            start_time_utc="2026-01-15T15:00Z", duration_minutes=60, priority="urgent"),
        ScheduledEvent(event_id="EVT-BOB-LATE", attendees=["Bob"],
            start_time_utc="2026-01-15T17:00Z", duration_minutes=60, priority="urgent"),
        ScheduledEvent(event_id="EVT-ALICE-MID", attendees=["Alice"],
            start_time_utc="2026-01-15T15:00Z", duration_minutes=60, priority="urgent"),
        ScheduledEvent(event_id="EVT-ALICE-BOB-URGENT", attendees=["Alice", "Bob"],
            start_time_utc="2026-01-15T14:00Z", duration_minutes=60, priority="urgent"),
        ScheduledEvent(event_id="EVT-ALICE-DEV-URGENT", attendees=["Alice", "Dev"],
            start_time_utc="2026-01-15T16:00Z", duration_minutes=60, priority="urgent"),
        ScheduledEvent(event_id="EVT-DEV-LOW", attendees=["Dev"],
            start_time_utc="2026-01-15T17:00Z", duration_minutes=60, priority="low",
            request_id="REQ-BUMPED-DEV"),
        ScheduledEvent(event_id="EVT-CEO-LATE", attendees=["CEO"],
            start_time_utc="2026-01-15T19:00Z", duration_minutes=120, priority="urgent"),
    ]
    bumped_dev = MeetingRequest(
        request_id="REQ-BUMPED-DEV", attendees=["Dev"], duration_minutes=60, priority="low",
        deadline_utc="2026-01-15T18:00Z", title="Dev Async Work",
    )
    req_urgent = MeetingRequest(
        request_id="REQ-URGENT-ALL-HANDS", attendees=["CEO", "Alice", "Bob"],
        duration_minutes=60, priority="urgent",
        deadline_utc="2026-01-15T15:00Z", title="Urgent All Hands",
    )
    req_cto = MeetingRequest(
        request_id="REQ-CTO-SYNC", attendees=["CEO", "CTO", "Alice"],
        duration_minutes=60, priority="medium",
        deadline_utc="2026-01-15T22:00Z", title="CTO Sync",
    )
    emergency = MeetingRequest(
        request_id="REQ-HARD-FU1", attendees=["CEO", "Alice"],
        duration_minutes=30, priority="urgent",
        deadline_utc="2026-01-15T23:00Z", title="Emergency Debrief",
    )
    return ScenarioSpec(
        scenario_id="HARD", description="The Zero-Sum Domino Cascade",
        current_time_utc="2026-01-15T08:00Z",
        participants=participants, calendar_state=calendar_state,
        pending_requests=[req_urgent, req_cto],
        all_requests=[req_urgent, req_cto, bumped_dev, emergency],
        max_turns=15, dynamic_followups=[emergency],
    )


def scenario_hard_b() -> ScenarioSpec:
    """HARD_B: The VIP No-Bump Gridlock.
    A multiday executive scheduling puzzle. Sprint Close must happen on day 1,
    but Alice's training occupies the only feasible day-1 slot. Board Prep
    must happen on day 2 before the CEO's investor review. The agent must
    inspect, move the training block across days, then place both requests in
    the correct order.

    Preferred hours are HIDDEN (HARD tier). Optimal: inspect all principals,
    reschedule Alice's blocker to day 2, place Sprint Close on day 1, then
    place Board Prep at the unique day-2 pre-investor slot.
    """
    participants = {
        "CEO": Participant(
            name="CEO", timezone="EST",
            working_hours=["08:00-17:00"],
            preferred_hours=["09:00-10:00"],
        ),
        "Alice": Participant(
            name="Alice", timezone="EST",
            working_hours=["09:00-17:00"],
            preferred_hours=["11:00-13:00"],
        ),
        "Bob": Participant(
            name="Bob", timezone="GMT",
            working_hours=["08:00-18:00"],
            preferred_hours=["16:00-17:00"],
        ),
        "Legal": Participant(
            name="Legal", timezone="UTC",
            working_hours=["16:00-17:00"],
            preferred_hours=["16:00-17:00"],
        ),
    }
    calendar_state = [
        ScheduledEvent(
            event_id="EVT-CEO-VIP", attendees=["CEO"],
            start_time_utc="2026-01-15T14:00Z", duration_minutes=180, priority="urgent",
        ),
        ScheduledEvent(
            event_id="EVT-CEO-INVESTOR", attendees=["CEO"],
            start_time_utc="2026-01-16T15:00Z", duration_minutes=120, priority="urgent",
        ),
        ScheduledEvent(
            event_id="EVT-ALICE-TRAIN", attendees=["Alice"],
            start_time_utc="2026-01-15T16:00Z", duration_minutes=120, priority="high",
        ),
        ScheduledEvent(
            event_id="EVT-BOB-CLIENT", attendees=["Bob"],
            start_time_utc="2026-01-15T14:00Z", duration_minutes=120, priority="medium",
        ),
    ]
    req1 = MeetingRequest(
        request_id="REQ-HARDB-1", attendees=["CEO", "Alice", "Bob"],
        duration_minutes=60, priority="urgent",
        deadline_utc="2026-01-16T15:00Z", title="Board Prep",
    )
    req2 = MeetingRequest(
        request_id="REQ-HARDB-2", attendees=["Alice", "Bob", "Legal"],
        duration_minutes=60, priority="high",
        deadline_utc="2026-01-15T17:00Z", title="Sprint Close",
    )
    return ScenarioSpec(
        scenario_id="HARD_B", description="The VIP No-Bump Gridlock",
        current_time_utc="2026-01-15T12:00Z",
        participants=participants, calendar_state=calendar_state,
        pending_requests=[req1, req2], all_requests=[req1, req2],
        max_turns=14,
    )


def scenario_hard_c() -> ScenarioSpec:
    """HARD_C: The Decoy Trap.
    A multiday downstream-consequence puzzle. The urgent real review has only
    one valid slot on day 2. The tempting high-priority trap request can also
    fit in that slot, so greedily taking the prettier day-2 placement ruins the
    urgent request. The right move is to bump the day-1 decoy, place the trap
    on day 1, reserve the unique day-2 slot for the urgent review, then recover
    the bumped placeholder later.

    Preferred hours are HIDDEN (HARD tier).
    """
    participants = {
        "Alice": Participant(
            name="Alice", timezone="EST",
            working_hours=["09:00-17:00"],
            preferred_hours=["10:00-12:00"],
        ),
        "Bob": Participant(
            name="Bob", timezone="UTC",
            working_hours=["09:00-17:00"],
            preferred_hours=["14:00-16:00"],
        ),
        "Dev": Participant(
            name="Dev", timezone="IST",
            working_hours=["14:00-23:00"],  # 08:30–17:30 UTC
            preferred_hours=["18:00-20:00"],  # 12:30–14:30 UTC
        ),
    }
    decoy = ScheduledEvent(
        event_id="EVT-DECOY", attendees=["Alice", "Bob"],
        start_time_utc="2026-01-15T15:00Z", duration_minutes=60, priority="low",
        request_id="REQ-DECOY",
    )
    bob_day1_early = ScheduledEvent(
        event_id="EVT-BOB-EARLY", attendees=["Bob"],
        start_time_utc="2026-01-15T13:00Z", duration_minutes=120, priority="urgent",
    )
    blocker = ScheduledEvent(
        event_id="EVT-BLOCKER", attendees=["Dev"],
        start_time_utc="2026-01-15T14:00Z", duration_minutes=120, priority="urgent",
    )
    bob_day1_block = ScheduledEvent(
        event_id="EVT-BOB-LATE", attendees=["Bob"],
        start_time_utc="2026-01-15T16:00Z", duration_minutes=60, priority="urgent",
    )
    alice_day2_block = ScheduledEvent(
        event_id="EVT-ALICE-DAY2", attendees=["Alice"],
        start_time_utc="2026-01-16T15:00Z", duration_minutes=120, priority="high",
    )
    req_trap = MeetingRequest(
        request_id="REQ-HARDC-TRAP", attendees=["Alice", "Bob"],
        duration_minutes=60, priority="high",
        deadline_utc="2026-01-15T16:00Z", title="Quick Sync",
    )
    req_real = MeetingRequest(
        request_id="REQ-HARDC-REAL", attendees=["Alice", "Bob", "Dev"],
        duration_minutes=60, priority="urgent",
        deadline_utc="2026-01-16T15:00Z", title="Critical Review",
    )
    req_decoy = MeetingRequest(
        request_id="REQ-DECOY", attendees=["Alice", "Bob"],
        duration_minutes=60, priority="low",
        deadline_utc="2026-01-16T18:00Z", title="Placeholder",
    )
    return ScenarioSpec(
        scenario_id="HARD_C", description="The Decoy Trap",
        current_time_utc="2026-01-15T12:00Z",
        participants=participants,
        calendar_state=[decoy, bob_day1_early, blocker, bob_day1_block, alice_day2_block],
        pending_requests=[req_trap, req_real],
        all_requests=[req_trap, req_real, req_decoy],
        max_turns=14,
    )


# ── Registry ───────────────────────────────────────────────────────────

_SCENARIO_REGISTRY: Dict[str, callable] = {
    "EASY":     scenario_easy,
    "EASY_B":   scenario_easy_b,
    "EASY_C":   scenario_easy_c,
    "MEDIUM":   scenario_medium,
    "MEDIUM_B": scenario_medium_b,
    "MEDIUM_C": scenario_medium_c,
    "HARD":     scenario_hard,
    "HARD_B":   scenario_hard_b,
    "HARD_C":   scenario_hard_c,
}

# Round-robin order used when no explicit scenario_id is given
_ROTATION_ORDER = ["EASY", "MEDIUM", "HARD"]


def get_scenario(scenario_id: Optional[str] = None, reset_count: int = 1) -> ScenarioSpec:
    """Load a scenario by ID, or pick one via round-robin rotation.

    Args:
        scenario_id: Explicit scenario key (e.g. "HARD_C"). Case-insensitive.
                     If None, the scenario is chosen by round-robin from
                     [EASY, MEDIUM, HARD] based on reset_count.
        reset_count: Episode counter used for round-robin when no ID given.

    Returns:
        A ScenarioSpec instance ready to be passed to the environment.

    Raises:
        KeyError: If scenario_id is provided but not found in the registry.
    """
    if scenario_id:
        key = scenario_id.strip().upper()
        if key not in _SCENARIO_REGISTRY:
            available = ", ".join(sorted(_SCENARIO_REGISTRY.keys()))
            raise KeyError(f"Unknown scenario '{scenario_id}'. Available: {available}")
        return _SCENARIO_REGISTRY[key]()

    # Round-robin among base tiers (EASY / MEDIUM / HARD)
    idx = (reset_count - 1) % len(_ROTATION_ORDER)
    return _SCENARIO_REGISTRY[_ROTATION_ORDER[idx]]()


def list_scenarios() -> Dict[str, str]:
    """Return a mapping of scenario_id → description for all registered scenarios."""
    return {key: fn().description for key, fn in _SCENARIO_REGISTRY.items()}
