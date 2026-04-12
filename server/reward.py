"""Decomposed reward computation for Meeting Negotiator V1.

9 named components, max 1.00:

  1. COMPLETION        (0.35) — all requests scheduled
  2. DEADLINE          (0.20) — meetings finish before deadline
  3. WORKING_HOURS     (0.15) — all within working hours
  4. PREFERENCE        (0.10) — preferred hours respected
  5. CONFLICTS         (0.10) — no double-bookings
  6. EFFICIENCY        (0.05) — low step count
  7. INVESTIGATION     (0.05) — investigated before acting (HARD only)
  8. STABILITY         (penalty) — harmful transitions reduce score
  9. RECOVERY          (bonus) — explicit recovery wins back some trust

All deterministic. No model loading. Perfect run → 0.99.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, time
from typing import Dict, List, Optional, Tuple, Any


PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "urgent": 3}

# ── Per-Component Weights ──────────────────────────────────────────────
W_COMPLETION = 0.35
W_DEADLINE = 0.20
W_WORKING_HOURS = 0.15
W_PREFERENCE = 0.10
W_CONFLICTS = 0.10
W_EFFICIENCY = 0.05
W_INVESTIGATION = 0.05

# ── Step reward constants ──────────────────────────────────────────────
STEP_PENALTY = 0.01
INVALID_ACTION_PENALTY = 0.05
CONFLICT_ACTION_PENALTY = 0.05
PREFERENCE_PENALTY_PER_ATTENDEE = 0.01

SCHEDULE_SUCCESS_REWARD = 0.25
RESCHEDULE_SUCCESS_REWARD = 0.10
PRIORITY_BONUS = 0.10
DEADLINE_PROXIMITY_BONUS = 0.05


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)


def _tz_offset(tz: str) -> timezone:
    tz_map = {"PST": -8, "CST": -6, "EST": -5, "GMT": 0, "UTC": 0, "IST": 5.5}
    if tz in tz_map:
        return timezone(timedelta(hours=tz_map[tz]))
    if tz.startswith("UTC"):
        raw = tz[3:]
        if not raw:
            return timezone.utc
        sign = 1
        if raw[0] == "+":
            sign = 1
            raw = raw[1:]
        elif raw[0] == "-":
            sign = -1
            raw = raw[1:]
        if ":" in raw:
            hours, minutes = raw.split(":", 1)
            offset = timedelta(hours=sign * int(hours), minutes=sign * int(minutes))
        else:
            offset = timedelta(hours=sign * int(raw))
        return timezone(offset)
    raise ValueError(f"unknown timezone {tz}")


def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def _within_blocks(local_start: datetime, local_end: datetime, blocks: List[str]) -> bool:
    days_span = (local_end.date() - local_start.date()).days
    for day_offset in range(days_span + 1):
        day = local_start.date() + timedelta(days=day_offset)
        for block in blocks:
            try:
                start_str, end_str = block.split("-")
                start_time = _parse_time(start_str)
                end_time = _parse_time(end_str)
            except ValueError:
                continue
            block_start = datetime.combine(day, start_time, tzinfo=local_start.tzinfo)
            block_end = datetime.combine(day, end_time, tzinfo=local_start.tzinfo)
            if block_end <= block_start:
                block_end += timedelta(days=1)
            if local_start >= block_start and local_end <= block_end:
                return True
    return False


def _within_working_hours(participant_data: Dict, start_dt: datetime, end_dt: datetime) -> bool:
    try:
        tz = _tz_offset(participant_data.get("timezone", "UTC"))
    except ValueError:
        return False
    local_start = start_dt.astimezone(tz)
    local_end = end_dt.astimezone(tz)
    return _within_blocks(local_start, local_end, participant_data.get("working_hours", []))


def _within_preferred_hours(participant_data: Dict, start_dt: datetime, end_dt: datetime) -> bool:
    pref = participant_data.get("preferred_hours", [])
    if not pref:
        return True
    try:
        tz = _tz_offset(participant_data.get("timezone", "UTC"))
    except ValueError:
        return False
    local_start = start_dt.astimezone(tz)
    local_end = end_dt.astimezone(tz)
    return _within_blocks(local_start, local_end, pref)


def compute_final_score(
    all_requests: List[Dict],
    calendar_state: List[Dict],
    participants: Dict[str, Dict],
    turn_count: int,
    max_turns: int,
    inspected_participants: Optional[List[str]] = None,
    scenario_id: str = "",
    system_state: str = "stable",
    escalation_budget_remaining: int = 0,
    triggered_followups: Optional[List[str]] = None,
    resolved_recovery_requests: Optional[List[str]] = None,
    modified_existing_events: Optional[List[Dict]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute the final episode score with a full decomposition.

    Returns:
        (score, breakdown_dict) where score is in [0.01, 0.99] and
        breakdown_dict maps component names to their individual contributions.
    """
    # Build lookup of scheduled events by request_id
    scheduled_by_request: Dict[str, Dict] = {}
    for event in calendar_state:
        req_id = event.get("request_id")
        if req_id:
            scheduled_by_request[req_id] = event

    total_requests = len(all_requests)
    if total_requests == 0:
        return 0.50, {"completion": 0.50}

    # ── 1. COMPLETION ──────────────────────────────────────────────
    scheduled_count = sum(1 for r in all_requests if r.get("request_id") in scheduled_by_request)
    completion = W_COMPLETION * (scheduled_count / total_requests)

    # ── 2. DEADLINE COMPLIANCE ─────────────────────────────────────
    deadline_violations = 0
    for request in all_requests:
        event = scheduled_by_request.get(request.get("request_id"))
        if event is None:
            continue
        start_dt = _parse_utc(event["start_time_utc"])
        end_dt = start_dt + timedelta(minutes=event["duration_minutes"])
        if end_dt > _parse_utc(request["deadline_utc"]):
            deadline_violations += 1
    if scheduled_count > 0:
        deadline = W_DEADLINE * (1.0 - deadline_violations / scheduled_count)
    else:
        deadline = 0.0

    scored_existing_events = modified_existing_events or []

    # ── 3. WORKING HOURS COMPLIANCE ────────────────────────────────
    wh_violations = 0
    wh_checks = 0
    for request in all_requests:
        event = scheduled_by_request.get(request.get("request_id"))
        if event is None:
            continue
        start_dt = _parse_utc(event["start_time_utc"])
        end_dt = start_dt + timedelta(minutes=event["duration_minutes"])
        for attendee in request.get("attendees", []):
            wh_checks += 1
            p = participants.get(attendee)
            if p is None:
                wh_violations += 1
                continue
            p_data = p if isinstance(p, dict) else p.model_dump() if hasattr(p, "model_dump") else {}
            try:
                if not _within_working_hours(p_data, start_dt, end_dt):
                    wh_violations += 1
            except ValueError:
                wh_violations += 1
    for event in scored_existing_events:
        start_dt = _parse_utc(event["start_time_utc"])
        end_dt = start_dt + timedelta(minutes=event["duration_minutes"])
        for attendee in event.get("attendees", []):
            wh_checks += 1
            p = participants.get(attendee)
            if p is None:
                wh_violations += 1
                continue
            p_data = p if isinstance(p, dict) else p.model_dump() if hasattr(p, "model_dump") else {}
            try:
                if not _within_working_hours(p_data, start_dt, end_dt):
                    wh_violations += 1
            except ValueError:
                wh_violations += 1
    if wh_checks > 0:
        working_hours = W_WORKING_HOURS * (1.0 - wh_violations / wh_checks)
    else:
        working_hours = W_WORKING_HOURS

    # ── 4. PREFERENCE QUALITY ──────────────────────────────────────
    pref_violations = 0
    pref_checks = 0
    for request in all_requests:
        event = scheduled_by_request.get(request.get("request_id"))
        if event is None:
            continue
        start_dt = _parse_utc(event["start_time_utc"])
        end_dt = start_dt + timedelta(minutes=event["duration_minutes"])
        for attendee in request.get("attendees", []):
            p = participants.get(attendee)
            if p is None:
                continue
            p_data = p if isinstance(p, dict) else p.model_dump() if hasattr(p, "model_dump") else {}
            if p_data.get("preferred_hours"):
                pref_checks += 1
                if not _within_preferred_hours(p_data, start_dt, end_dt):
                    pref_violations += 1
    for event in scored_existing_events:
        start_dt = _parse_utc(event["start_time_utc"])
        end_dt = start_dt + timedelta(minutes=event["duration_minutes"])
        for attendee in event.get("attendees", []):
            p = participants.get(attendee)
            if p is None:
                continue
            p_data = p if isinstance(p, dict) else p.model_dump() if hasattr(p, "model_dump") else {}
            if p_data.get("preferred_hours"):
                pref_checks += 1
                if not _within_preferred_hours(p_data, start_dt, end_dt):
                    pref_violations += 1
    if pref_checks > 0:
        preference = W_PREFERENCE * (1.0 - pref_violations / pref_checks)
    else:
        preference = W_PREFERENCE

    # ── 5. CONFLICT AVOIDANCE ──────────────────────────────────────
    conflict_count = 0
    events = list(calendar_state)
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            a = events[i]
            b = events[j]
            if not set(a.get("attendees", [])).intersection(b.get("attendees", [])):
                continue
            a_start = _parse_utc(a["start_time_utc"])
            a_end = a_start + timedelta(minutes=a["duration_minutes"])
            b_start = _parse_utc(b["start_time_utc"])
            b_end = b_start + timedelta(minutes=b["duration_minutes"])
            if max(a_start, b_start) < min(a_end, b_end):
                conflict_count += 1
    conflicts = W_CONFLICTS * max(0.0, 1.0 - conflict_count * 0.5)

    # ── 6. EFFICIENCY ──────────────────────────────────────────────
    if max_turns > 0:
        ratio = min(1.0, (max_turns - turn_count) / max_turns)
        efficiency = W_EFFICIENCY * max(0.0, ratio)
    else:
        efficiency = 0.0

    # PATCH: Calculate completion ratio
    completion_ratio = scheduled_count / total_requests if total_requests > 0 else 0.0

    # Scale compliance scores by completion ratio
    deadline = deadline * completion_ratio
    working_hours = working_hours * completion_ratio
    preference = preference * completion_ratio
    conflicts = conflicts * completion_ratio

    
    # ── 7. INVESTIGATION DISCIPLINE (HARD only) ────────────────────
    if scenario_id.upper().startswith("HARD"):
        relevant_participants = set()
        for r in all_requests:
            relevant_participants.update(r.get("attendees", []))
        if relevant_participants and inspected_participants:
            inspected_ratio = len(set(inspected_participants).intersection(relevant_participants)) / len(relevant_participants)
            investigation = W_INVESTIGATION * inspected_ratio
        else:
            investigation = 0.0
    else:
        investigation = W_INVESTIGATION  # full credit for non-hard tasks

    # ── 8. STABILITY PENALTY ──────────────────────────────────────
    followup_count = len(triggered_followups or [])
    stability_penalty = 0.0
    if system_state == "strained":
        stability_penalty = 0.04
    elif system_state == "recovery_needed":
        stability_penalty = 0.07
    elif system_state == "escalated":
        stability_penalty = 0.10
    stability_penalty += min(0.06, followup_count * 0.02)
    if escalation_budget_remaining < 0:
        stability_penalty += min(0.04, abs(escalation_budget_remaining) * 0.02)

    # ── 9. RECOVERY CREDIT ────────────────────────────────────────
    recovery_credit = min(0.05, len(resolved_recovery_requests or []) * 0.025)

    # ── TOTAL ──────────────────────────────────────────────────────
    breakdown = {
        "completion": round(completion, 4),
        "deadline_compliance": round(deadline, 4),
        "working_hours_compliance": round(working_hours, 4),
        "preference_quality": round(preference, 4),
        "conflict_avoidance": round(conflicts, 4),
        "efficiency": round(efficiency, 4),
        "investigation_discipline": round(investigation, 4),
        "stability_penalty": round(-stability_penalty, 4),
        "recovery_credit": round(recovery_credit, 4),
    }

    total = (
        completion + deadline + working_hours + preference
        + conflicts + efficiency + investigation
        - stability_penalty + recovery_credit
    )
    score = round(min(0.99, max(0.01, total)), 4)

    return score, breakdown
