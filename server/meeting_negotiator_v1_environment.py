# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Meeting Negotiator V1 Environment Implementation.
A multi-step constraint satisfaction RL environment for calendar coordination.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        MeetingNegotiatorV1Action,
        MeetingNegotiatorV1Observation,
        MeetingNegotiatorV1State,
        Participant,
        MeetingRequest,
        ScheduledEvent,
    )
except ImportError:
    import os as _os
    import sys as _sys
    _parent_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _parent_dir not in _sys.path:
        _sys.path.insert(0, _parent_dir)
    from models import (
        MeetingNegotiatorV1Action,
        MeetingNegotiatorV1Observation,
        MeetingNegotiatorV1State,
        Participant,
        MeetingRequest,
        ScheduledEvent,
    )


PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "urgent": 3}

STEP_PENALTY = 0.01
INVALID_ACTION_PENALTY = 0.05
CONFLICT_ACTION_PENALTY = 0.05
PREFERENCE_PENALTY_PER_ATTENDEE = 0.01

SCHEDULE_SUCCESS_REWARD    = 0.25   # agent successfully books a pending request
RESCHEDULE_SUCCESS_REWARD  = 0.10   # agent successfully moves an existing event
PRIORITY_BONUS             = 0.10   # extra credit when booking urgent/high priority
DEADLINE_PROXIMITY_BONUS   = 0.05   # extra credit when booked well before deadline

SCORE_PENALTIES = {
    "unscheduled": 0.40,
    "deadline": 0.20,
    "working_hours": 0.25,
    "conflict": 0.30,
    "preference": 0.05,
    "priority_inversion": 0.15,
}


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    description: str
    current_time_utc: str
    participants: Dict[str, Participant]
    calendar_state: List[ScheduledEvent]
    pending_requests: List[MeetingRequest]
    all_requests: List[MeetingRequest]
    max_turns: int = 15


class MeetingNegotiatorV1Environment(Environment):
    """
    The Meeting Negotiator Environment.

    The agent must schedule pending meeting requests into a calendar without
    creating conflicts, while respecting working hours, time zones, priorities,
    and deadlines.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = MeetingNegotiatorV1State()
        self._reset_count = 0

    def reset(self, scenario_id: Optional[str] = None) -> MeetingNegotiatorV1Observation:
        self._reset_count += 1
        scenario = self._select_scenario(self._reset_count, scenario_id)

        self._state = MeetingNegotiatorV1State(
            episode_id=str(uuid4()),
            step_count=0,
            current_time_utc=scenario.current_time_utc,
            participants=scenario.participants,
            calendar_state=list(scenario.calendar_state),
            pending_requests=list(scenario.pending_requests),
            all_requests=list(scenario.all_requests),
            last_action_feedback=f"Scenario loaded: {scenario.scenario_id} - {scenario.description}",
            scenario_id=scenario.scenario_id,
            max_turns=scenario.max_turns,
            turn_count=0,
            is_done=False,
            total_reward=0.0,
            score=None,
        )

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: MeetingNegotiatorV1Action) -> MeetingNegotiatorV1Observation:  # type: ignore[override]
        if self._state.is_done:
            return self._build_observation(reward=0.0, done=True)

        self._state.turn_count += 1

        reward = -STEP_PENALTY
        feedback = ""
        done = False

        if self._state.turn_count >= self._state.max_turns and action.command != "SubmitFinalCalendar":
            self._state.is_done = True
            self._state.score = 0.01
            feedback = "Max turns exceeded without submission. Episode failed."
            reward -= INVALID_ACTION_PENALTY
            self._state.total_reward += reward
            self._state.last_action_feedback = feedback
            return self._build_observation(reward=reward, done=True)

        if action.command == "CheckAvailability":
            reward, feedback = self._handle_check_availability(action, reward)
        elif action.command == "ScheduleNew":
            reward, feedback = self._handle_schedule_new(action, reward)
        elif action.command == "RescheduleExisting":
            reward, feedback = self._handle_reschedule_existing(action, reward)
        elif action.command == "SubmitFinalCalendar":
            score = self._compute_score()
            self._state.score = score
            self._state.is_done = True
            done = True
            reward += score
            feedback = f"Final submission received. Score={score:.2f}."
        else:
            reward -= INVALID_ACTION_PENALTY
            feedback = f"Unknown command: {action.command}."

        self._state.total_reward += reward
        self._state.last_action_feedback = feedback

        return self._build_observation(reward=reward, done=done or self._state.is_done)

    @property
    def state(self) -> MeetingNegotiatorV1State:
        return self._state

    # Scenario setup

    def _select_scenario(self, reset_count: int, scenario_id: Optional[str]) -> ScenarioSpec:
        scenarios = {
            "EASY": self._scenario_easy(),
            "MEDIUM": self._scenario_medium(),
            "HARD": self._scenario_hard(),
        }
        if scenario_id:
            key = scenario_id.strip().upper()
            if key in scenarios:
                return scenarios[key]
        order = ["EASY", "MEDIUM", "HARD"]
        # order = ["HARD"]
        index = (reset_count - 1) % len(order)
        return scenarios[order[index]]

    def _scenario_easy(self) -> ScenarioSpec:
        participants = {
            "Alice": Participant(
                name="Alice",
                timezone="UTC",
                working_hours=["09:00-12:00", "13:00-17:00"],
                preferred_hours=["10:00-11:30", "14:00-16:00"],
            ),
            "Bob": Participant(
                name="Bob",
                timezone="UTC",
                working_hours=["09:00-12:00", "13:00-17:00"],
                preferred_hours=["09:30-11:00", "15:00-16:30"],
            ),
        }
        request = MeetingRequest(
            request_id="REQ-EASY-1",
            attendees=["Alice", "Bob"],
            duration_minutes=60,
            priority="medium",
            deadline_utc="2026-01-15T17:00Z",
            title="1:1 Alice/Bob",
        )
        return ScenarioSpec(
            scenario_id="EASY",
            description="The Empty Slate",
            current_time_utc="2026-01-15T08:00Z",
            participants=participants,
            calendar_state=[],
            pending_requests=[request],
            all_requests=[request],
            max_turns=9,
        )

    def _scenario_medium(self) -> ScenarioSpec:
        participants = {
            "Priya": Participant(
                name="Priya",
                timezone="GMT",
                working_hours=["09:00-18:00"],
                preferred_hours=["16:00-17:00"], 
            ),
            "Jordan": Participant(
                name="Jordan",
                timezone="EST",
                working_hours=["09:00-17:00"],   # 14:00Z - 22:00Z
                preferred_hours=["11:00-12:00"], # 16:00Z - 17:00Z
            ),
            "Alex": Participant(
                name="Alex",
                timezone="PST",
                working_hours=["08:00-17:00"],   #16:00Z - 1:00Z
                preferred_hours=["09:00-10:00"], # 17:00Z - 18:00Z
            ),
        }

        calendar_state = [
            ScheduledEvent(
                event_id="EVT-PRIYA-BLOCK",
                attendees=["Priya"],
                start_time_utc="2026-01-15T09:00Z",
                duration_minutes=420,                   #09:00Z - 16:00Z
                priority="medium",
            )
        ]

        req_1 = MeetingRequest(
            request_id="REQ-MED-1",
            attendees=["Priya", "Jordan", "Alex"],
            duration_minutes=60,
            priority="high",
            deadline_utc="2026-01-15T18:00Z",
            title="Full Team Sync",
        )
        
        req_2 = MeetingRequest(
            request_id="REQ-MED-2",
            attendees=["Priya", "Jordan"],
            duration_minutes=60,
            priority="medium",
            deadline_utc="2026-01-15T18:00Z",
            title="Priya-Jordan Handoff",
        )

        return ScenarioSpec(
            scenario_id="MEDIUM",
            description="The Greedy Preference Trap",
            current_time_utc="2026-01-15T15:00Z",
            participants=participants,
            calendar_state=calendar_state,
            pending_requests=[req_1, req_2],
            all_requests=[req_1, req_2],
            max_turns=10,
        )

    def _scenario_hard(self) -> ScenarioSpec:
        # UTC Working Hours Math:
        # CEO (EST): 08:00-17:00 EST -> 13:00Z-22:00Z
        # CTO (PST): 09:00-18:00 PST -> 17:00Z-02:00Z
        # Alice (EST): 09:00-17:00 EST -> 14:00Z-22:00Z
        # Bob (GMT): 08:00-18:00 GMT -> 08:00Z-18:00Z
        # Dev (IST): 14:00-23:59 IST -> 08:30Z-18:29Z (Night shift to overlap US)

        participants = {
            "CEO": Participant(
                name="CEO",
                timezone="EST",
                working_hours=["08:00-17:00"],
                preferred_hours=["16:00-17:00"], # 21:00Z-22:00Z (The Global Trap Optimal Slot)
            ),
            "CTO": Participant(
                name="CTO",
                timezone="PST",
                working_hours=["09:00-18:00"],
                preferred_hours=["10:00-11:00"], # 18:00Z-19:00Z (The Global Trap Bait Slot)
            ),
            "Alice": Participant(
                name="Alice",
                timezone="EST",
                working_hours=["09:00-17:00"],
                preferred_hours=["16:00-17:00"], # 21:00Z-22:00Z (The Global Trap Optimal Slot)
            ),
            "Bob": Participant(
                name="Bob",
                timezone="GMT",
                working_hours=["08:00-18:00"],
                preferred_hours=["16:00-17:00"], 
            ),
            "Dev": Participant(
                name="Dev",
                timezone="IST",
                working_hours=["14:00-23:59"], 
                preferred_hours=["14:30-16:30"], # 09:00Z-11:00Z (Morning free slot for bumped meeting)
            ),
        }

        calendar_state = [
            # === BLOCKERS TO FORCE THE CASCADE ===
            # Bob is completely blocked except for 16:00Z
            ScheduledEvent(
                event_id="EVT-BOB-MORNING",
                attendees=["Bob"],
                start_time_utc="2026-01-15T08:00Z",
                duration_minutes=360, # 08:00Z - 14:00Z
                priority="urgent",
            ),
            ScheduledEvent(
                event_id="EVT-BOB-MID",
                attendees=["Bob"],
                start_time_utc="2026-01-15T15:00Z",
                duration_minutes=60,
                priority="urgent",
            ),
            # THE FIX: "The Door Slammer" - Blocks Bob from jumping straight to 17:00Z
            ScheduledEvent(
                event_id="EVT-BOB-LATE",
                attendees=["Bob"],
                start_time_utc="2026-01-15T17:00Z",
                duration_minutes=60,
                priority="urgent",
            ),
            # Alice is blocked at 15:00Z
            ScheduledEvent(
                event_id="EVT-ALICE-MID",
                attendees=["Alice"],
                start_time_utc="2026-01-15T15:00Z",
                duration_minutes=60,
                priority="urgent",
            ),

            # === PHASE 1: THE TRIPLE DOMINO ===
            # Domino 1: Blocks 14:00Z. MUST move to 16:00Z (only open slot left for Alice+Bob).
            ScheduledEvent(
                event_id="EVT-ALICE-BOB-URGENT",
                attendees=["Alice", "Bob"],
                start_time_utc="2026-01-15T14:00Z",
                duration_minutes=60,
                priority="urgent",
            ),
            # Domino 2: Blocks 16:00Z. MUST move to 17:00Z.
            ScheduledEvent(
                event_id="EVT-ALICE-DEV-URGENT", 
                attendees=["Alice", "Dev"],
                start_time_utc="2026-01-15T16:00Z",
                duration_minutes=60,
                priority="urgent", 
            ),
            # Domino 3: The Bump Target at 17:00Z. Gets destroyed by Domino 2.
            ScheduledEvent(
                event_id="EVT-DEV-LOW",
                attendees=["Dev"],
                start_time_utc="2026-01-15T17:00Z",
                duration_minutes=60,
                priority="low",
                request_id="REQ-BUMPED-DEV",
            ),

            # === PHASE 2: THE GLOBAL PENALTY TRAP ===
            # Blocks 19:00Z and 20:00Z to force the Trolley Problem (18:00Z vs 21:00Z)
            ScheduledEvent(
                event_id="EVT-CEO-LATE",
                attendees=["CEO"],
                start_time_utc="2026-01-15T19:00Z",
                duration_minutes=120, # 19:00Z - 21:00Z
                priority="urgent",
            ),
        ]

        bumped_dev = MeetingRequest(
            request_id="REQ-BUMPED-DEV",
            attendees=["Dev"],
            duration_minutes=60,
            priority="low",
            deadline_utc="2026-01-15T18:00Z",
            title="Dev Async Work",
        )

        req_urgent = MeetingRequest(
            request_id="REQ-URGENT-ALL-HANDS",
            attendees=["CEO", "Alice", "Bob"],
            duration_minutes=60,
            priority="urgent",
            deadline_utc="2026-01-15T15:00Z", # Forces the meeting into exactly 14:00Z
            title="Urgent All Hands",
        )

        req_cto = MeetingRequest(
            request_id="REQ-CTO-SYNC",
            attendees=["CEO", "CTO", "Alice"],
            duration_minutes=60,
            priority="medium",
            deadline_utc="2026-01-15T22:00Z",
            title="CTO Sync",
        )

        return ScenarioSpec(
            scenario_id="HARD",
            description="The Zero-Sum Domino Cascade",
            current_time_utc="2026-01-15T08:00Z",
            participants=participants,
            calendar_state=calendar_state,
            pending_requests=[req_urgent, req_cto],
            all_requests=[req_urgent, req_cto, bumped_dev],
            max_turns=12,
        )

    # Action handlers

    def _handle_check_availability(
        self, action: MeetingNegotiatorV1Action, reward: float
    ) -> Tuple[float, str]:
        if not action.target_id or not action.proposed_start_utc:
            return reward - INVALID_ACTION_PENALTY, "CheckAvailability requires target_id and proposed_start_utc."

        request = self._find_request(action.target_id)
        if request is None:
            return reward - INVALID_ACTION_PENALTY, f"Request {action.target_id} not found."

        start_dt = self._parse_utc(action.proposed_start_utc)
        end_dt = start_dt + timedelta(minutes=request.duration_minutes)

        issues, conflicts = self._evaluate_constraints(request, start_dt, end_dt)
        if conflicts:
            conflict_names = ", ".join([c.event_id for c in conflicts])
            issues.append(f"conflicts: {conflict_names}")

        if issues:
            return reward, f"Slot NOT available: {', '.join(issues)}."
        return reward, "Slot available for all attendees."

    def _handle_schedule_new(
        self, action: MeetingNegotiatorV1Action, reward: float
    ) -> Tuple[float, str]:
        if not action.target_id or not action.proposed_start_utc:
            return reward - INVALID_ACTION_PENALTY, "ScheduleNew requires target_id and proposed_start_utc."

        request = self._find_request(action.target_id)
        if request is None:
            return reward - INVALID_ACTION_PENALTY, f"Request {action.target_id} not found in pending requests."

        start_dt = self._parse_utc(action.proposed_start_utc)
        end_dt = start_dt + timedelta(minutes=request.duration_minutes)

        issues, conflicts = self._evaluate_constraints(request, start_dt, end_dt)
        if issues:
            reward -= INVALID_ACTION_PENALTY
            return reward, f"Schedule failed: {', '.join(issues)}."

        blocking = self._blocking_conflicts(conflicts, request.priority)
        if blocking:
            reward -= CONFLICT_ACTION_PENALTY
            conflict_ids = ", ".join([c.event_id for c in blocking])
            return reward, f"Schedule failed due to higher/equal priority conflicts: {conflict_ids}."

        bumped = self._bump_conflicts(conflicts, request.priority)

        new_event = ScheduledEvent(
            event_id=f"EVT-{uuid4().hex[:8]}",
            attendees=request.attendees,
            start_time_utc=self._format_utc(start_dt),
            duration_minutes=request.duration_minutes,
            priority=request.priority,
            request_id=request.request_id,
        )
        self._state.calendar_state.append(new_event)
        self._state.pending_requests = [r for r in self._state.pending_requests if r.request_id != request.request_id]

        reward += SCHEDULE_SUCCESS_REWARD

        # bonus for priority — urgent meetings matter more than low ones
        priority_level = PRIORITY_ORDER[request.priority]  # 0=low, 3=urgent
        reward += PRIORITY_BONUS * (priority_level / 3)

        # bonus for scheduling comfortably before the deadline
        deadline_dt = self._parse_utc(request.deadline_utc)
        minutes_remaining = (deadline_dt - end_dt).total_seconds() / 60
        if minutes_remaining > 30:
            reward += DEADLINE_PROXIMITY_BONUS


        feedback = f"Scheduled {request.request_id} at {new_event.start_time_utc}."
        if bumped:
            feedback += f" Bumped lower-priority events: {', '.join(bumped)}."

        return round(reward, 4), feedback

    def _handle_reschedule_existing(
        self, action: MeetingNegotiatorV1Action, reward: float
    ) -> Tuple[float, str]:
        if not action.target_id or not action.proposed_start_utc:
            return reward - INVALID_ACTION_PENALTY, "RescheduleExisting requires target_id and proposed_start_utc."

        event = self._find_event(action.target_id)
        if event is None:
            return reward - INVALID_ACTION_PENALTY, f"Event {action.target_id} not found."

        start_dt = self._parse_utc(action.proposed_start_utc)
        end_dt = start_dt + timedelta(minutes=event.duration_minutes)

        temp_request = MeetingRequest(
            request_id=event.request_id or f"TEMP-{event.event_id}",
            attendees=event.attendees,
            duration_minutes=event.duration_minutes,
            priority=event.priority,
            deadline_utc=self._request_deadline(event.request_id),
            title="Reschedule",
        )

        issues, conflicts = self._evaluate_constraints(temp_request, start_dt, end_dt, ignore_event_id=event.event_id)
        if issues:
            reward -= INVALID_ACTION_PENALTY
            return reward, f"Reschedule failed: {', '.join(issues)}."

        blocking = self._blocking_conflicts(conflicts, event.priority)
        if blocking:
            reward -= CONFLICT_ACTION_PENALTY
            conflict_ids = ", ".join([c.event_id for c in blocking])
            return reward, f"Reschedule failed due to higher/equal priority conflicts: {conflict_ids}."

        bumped = self._bump_conflicts(conflicts, event.priority)
        event.start_time_utc = self._format_utc(start_dt)

        reward += RESCHEDULE_SUCCESS_REWARD

        feedback = f"Rescheduled {event.event_id} to {event.start_time_utc}."
        pref_violations = self._preference_violations(event.attendees, start_dt, end_dt)
        if pref_violations:
            reward -= PREFERENCE_PENALTY_PER_ATTENDEE * len(pref_violations)
            feedback += f" Outside preferred hours: {', '.join(pref_violations)}."
        if bumped:
            feedback += f" Bumped lower-priority events: {', '.join(bumped)}."

        return round(reward, 4), feedback

    # Constraint evaluation

    def _evaluate_constraints(
        self,
        request: MeetingRequest,
        start_dt: datetime,
        end_dt: datetime,
        ignore_event_id: Optional[str] = None,
    ) -> Tuple[List[str], List[ScheduledEvent]]:
        issues: List[str] = []

        if start_dt < self._parse_utc(self._state.current_time_utc):
            issues.append("proposed time is in the past")

        if request.deadline_utc and end_dt > self._parse_utc(request.deadline_utc):
            issues.append("misses deadline")

        for attendee in request.attendees:
            participant = self._state.participants.get(attendee)
            if participant is None:
                issues.append(f"unknown attendee {attendee}")
                continue
            try:
                if not self._within_working_hours(participant, start_dt, end_dt):
                    issues.append(f"{attendee} outside working hours")
            except ValueError as exc:
                issues.append(str(exc))

        conflicts = self._find_conflicts(request.attendees, start_dt, end_dt, ignore_event_id)

        return issues, conflicts

    def _find_conflicts(
        self,
        attendees: List[str],
        start_dt: datetime,
        end_dt: datetime,
        ignore_event_id: Optional[str] = None,
    ) -> List[ScheduledEvent]:
        conflicts = []
        for event in self._state.calendar_state:
            if ignore_event_id and event.event_id == ignore_event_id:
                continue
            if not set(attendees).intersection(event.attendees):
                continue
            ev_start = self._parse_utc(event.start_time_utc)
            ev_end = ev_start + timedelta(minutes=event.duration_minutes)
            if max(start_dt, ev_start) < min(end_dt, ev_end):
                conflicts.append(event)
        return conflicts

    def _blocking_conflicts(self, conflicts: List[ScheduledEvent], new_priority: str) -> List[ScheduledEvent]:
        blocking = []
        for event in conflicts:
            if PRIORITY_ORDER[event.priority] >= PRIORITY_ORDER[new_priority]:
                blocking.append(event)
        return blocking

    def _bump_conflicts(self, conflicts: List[ScheduledEvent], new_priority: str) -> List[str]:
        bumped_ids: List[str] = []
        if not conflicts:
            return bumped_ids

        for event in list(conflicts):
            if PRIORITY_ORDER[event.priority] < PRIORITY_ORDER[new_priority]:
                if event in self._state.calendar_state:
                    self._state.calendar_state.remove(event)
                bumped_ids.append(event.event_id)
                request = self._request_from_event(event)
                if request and not self._request_in_pending(request.request_id):
                    self._state.pending_requests.append(request)
        return bumped_ids

    def _request_from_event(self, event: ScheduledEvent) -> Optional[MeetingRequest]:
        if event.request_id:
            for req in self._state.all_requests:
                if req.request_id == event.request_id:
                    return req

        # Create new request for bumped event without original request_id
        new_request = MeetingRequest(
            request_id=f"REQ-BUMP-{event.event_id}",
            attendees=event.attendees,
            duration_minutes=event.duration_minutes,
            priority=event.priority,
            deadline_utc=self._format_utc(self._parse_utc(self._state.current_time_utc) + timedelta(days=1)),
            title="Bumped meeting",
        )

        # Add to all_requests so the grader will evaluate it
        if not any(r.request_id == new_request.request_id for r in self._state.all_requests):
            self._state.all_requests.append(new_request)
        return new_request

    def _request_in_pending(self, request_id: str) -> bool:
        return any(r.request_id == request_id for r in self._state.pending_requests)

    # Scoring

    def _compute_score(self) -> float:
        penalty = 0.0
        scheduled_by_request: Dict[str, ScheduledEvent] = {
            event.request_id: event for event in self._state.calendar_state if event.request_id
        }

        for request in self._state.all_requests:
            event = scheduled_by_request.get(request.request_id)
            if event is None:
                penalty += SCORE_PENALTIES["unscheduled"]
                continue

            start_dt = self._parse_utc(event.start_time_utc)
            end_dt = start_dt + timedelta(minutes=event.duration_minutes)

            if start_dt > self._parse_utc(request.deadline_utc):
                penalty += SCORE_PENALTIES["deadline"]

            for attendee in request.attendees:
                participant = self._state.participants.get(attendee)
                if participant is None:
                    penalty += SCORE_PENALTIES["working_hours"]
                    continue
                try:
                    if not self._within_working_hours(participant, start_dt, end_dt):
                        penalty += SCORE_PENALTIES["working_hours"]
                except ValueError:
                    penalty += SCORE_PENALTIES["working_hours"]
                if participant.preferred_hours:
                    if not self._within_preferred_hours(participant, start_dt, end_dt):
                        penalty += SCORE_PENALTIES["preference"]

        penalty += self._conflict_penalty()

        score = max(0.01, 1.0 - penalty)
        return min(0.99, score)

    def _conflict_penalty(self) -> float:
        penalty = 0.0
        events = list(self._state.calendar_state)
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                a = events[i]
                b = events[j]
                if not set(a.attendees).intersection(b.attendees):
                    continue
                a_start = self._parse_utc(a.start_time_utc)
                a_end = a_start + timedelta(minutes=a.duration_minutes)
                b_start = self._parse_utc(b.start_time_utc)
                b_end = b_start + timedelta(minutes=b.duration_minutes)
                if max(a_start, b_start) < min(a_end, b_end):
                    penalty += SCORE_PENALTIES["conflict"]
                    if PRIORITY_ORDER[a.priority] != PRIORITY_ORDER[b.priority]:
                        penalty += SCORE_PENALTIES["priority_inversion"]
        return penalty

    # Time helpers

    def _parse_utc(self, value: str) -> datetime:
        return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)

    def _format_utc(self, value: datetime) -> str:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    def _tz_offset(self, tz: str) -> timezone:
        tz_map = {"PST": -8, "EST": -5, "GMT": 0, "UTC": 0, "IST" : 5.5}
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

    def _preference_violations(
        self, attendees: List[str], start_dt: datetime, end_dt: datetime
    ) -> List[str]:
        violations: List[str] = []
        for attendee in attendees:
            participant = self._state.participants.get(attendee)
            if participant is None:
                continue
            if participant.preferred_hours and not self._within_preferred_hours(participant, start_dt, end_dt):
                violations.append(attendee)
        return violations

    def _within_working_hours(self, participant: Participant, start_dt: datetime, end_dt: datetime) -> bool:
        tz = self._tz_offset(participant.timezone)
        local_start = start_dt.astimezone(tz)
        local_end = end_dt.astimezone(tz)
        return self._within_blocks(local_start, local_end, participant.working_hours)

    def _within_preferred_hours(self, participant: Participant, start_dt: datetime, end_dt: datetime) -> bool:
        if not participant.preferred_hours:
            return True
        try:
            tz = self._tz_offset(participant.timezone)
        except ValueError:
            return False
        local_start = start_dt.astimezone(tz)
        local_end = end_dt.astimezone(tz)
        return self._within_blocks(local_start, local_end, participant.preferred_hours)

    def _within_blocks(self, local_start: datetime, local_end: datetime, blocks: List[str]) -> bool:
        days_span = (local_end.date() - local_start.date()).days
        for day_offset in range(days_span + 1):
            day = local_start.date() + timedelta(days=day_offset)
            for block in blocks:
                try:
                    start_str, end_str = block.split("-")
                    start_time = self._parse_time(start_str)
                    end_time = self._parse_time(end_str)
                except ValueError:
                    continue

                block_start = datetime.combine(day, start_time, tzinfo=local_start.tzinfo)
                block_end = datetime.combine(day, end_time, tzinfo=local_start.tzinfo)
                if block_end <= block_start:
                    block_end += timedelta(days=1)

                if local_start >= block_start and local_end <= block_end:
                    return True
        return False

    def _parse_time(self, value: str) -> time:
        return datetime.strptime(value, "%H:%M").time()

    # Utilities

    def _find_request(self, request_id: str) -> Optional[MeetingRequest]:
        for req in self._state.pending_requests:
            if req.request_id == request_id:
                return req
        return None

    def _find_event(self, event_id: str) -> Optional[ScheduledEvent]:
        for event in self._state.calendar_state:
            if event.event_id == event_id:
                return event
        return None

    def _request_deadline(self, request_id: Optional[str]) -> str:
        if request_id:
            for req in self._state.all_requests:
                if req.request_id == request_id:
                    return req.deadline_utc
        return self._format_utc(self._parse_utc(self._state.current_time_utc) + timedelta(days=1))

    def _build_observation(self, reward: float, done: bool) -> MeetingNegotiatorV1Observation:
        return MeetingNegotiatorV1Observation(
            current_time_utc=self._state.current_time_utc,
            participants=self._state.participants,
            calendar_state=self._state.calendar_state,
            pending_requests=self._state.pending_requests,
            last_action_feedback=self._state.last_action_feedback,
            turn_count=self._state.turn_count,
            max_turns=self._state.max_turns,
            score=self._state.score,
            reward=reward,
            done=done,
        )
