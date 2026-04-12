# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Meeting Negotiator V1 Environment Implementation.
A multi-step constraint satisfaction RL environment for calendar coordination.

Features:
  - 8 action types including investigation and undo
  - Partial observability on HARD (hidden preferences)
  - Seed-based anti-memorization randomization
  - Dynamic follow-up requests mid-episode
  - Decomposed reward transparency
  - 9 scenarios (3 per tier)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time
from typing import Dict, List, Optional, Tuple, Any
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
        VALID_COMMANDS,
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
        VALID_COMMANDS,
    )

try:
    from .reward import (
        STEP_PENALTY, INVALID_ACTION_PENALTY, CONFLICT_ACTION_PENALTY,
        PREFERENCE_PENALTY_PER_ATTENDEE, SCHEDULE_SUCCESS_REWARD,
        RESCHEDULE_SUCCESS_REWARD, PRIORITY_BONUS, DEADLINE_PROXIMITY_BONUS,
        compute_final_score,
    )
    from .scenario_resolver import resolve_scenario
    from .scenarios import ScenarioSpec, get_scenario
except ImportError:
    import os as _os2
    import sys as _sys2
    _server_dir = _os2.path.dirname(_os2.path.abspath(__file__))
    if _server_dir not in _sys2.path:
        _sys2.path.insert(0, _server_dir)
    from reward import (
        STEP_PENALTY, INVALID_ACTION_PENALTY, CONFLICT_ACTION_PENALTY,
        PREFERENCE_PENALTY_PER_ATTENDEE, SCHEDULE_SUCCESS_REWARD,
        RESCHEDULE_SUCCESS_REWARD, PRIORITY_BONUS, DEADLINE_PROXIMITY_BONUS,
        compute_final_score,
    )
    from scenario_resolver import resolve_scenario
    from scenarios import ScenarioSpec, get_scenario


PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "urgent": 3}

SCORE_PENALTIES = {
    "unscheduled": 0.40, "deadline": 0.20, "working_hours": 0.25,
    "conflict": 0.30, "preference": 0.05, "priority_inversion": 0.15,
}




class MeetingNegotiatorV1Environment(Environment):
    """The Meeting Negotiator Environment with expanded action space,
    partial observability, and decomposed reward transparency."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = MeetingNegotiatorV1State()
        self._reset_count = 0
        self._hidden_prefs: Dict[str, List[str]] = {}
        self._dynamic_followups: List[MeetingRequest] = []
        self._followup_trigger_step: int = 0
        self._protected_event_ids: List[str] = []
        self._recovery_specs: Dict[str, Dict[str, Any]] = {}

    def reset(self, scenario_id: Optional[str] = None, seed: Optional[int] = None) -> MeetingNegotiatorV1Observation:
        self._reset_count += 1
        scenario = self._select_scenario(self._reset_count, scenario_id)

        # Apply seed-based randomization
        if seed is not None:
            scenario_data = {
                "participants": {n: p.model_dump() for n, p in scenario.participants.items()},
                "calendar_state": [e.model_dump() for e in scenario.calendar_state],
                "pending_requests": [r.model_dump() for r in scenario.pending_requests],
                "all_requests": [r.model_dump() for r in scenario.all_requests],
            }
            resolved = resolve_scenario(scenario_data, seed)
            participants = {n: Participant(**p) for n, p in resolved["participants"].items()}
            calendar_state = [ScheduledEvent(**e) for e in resolved["calendar_state"]]
            pending_requests = [MeetingRequest(**r) for r in resolved["pending_requests"]]
            all_requests = [MeetingRequest(**r) for r in resolved["all_requests"]]
        else:
            participants = scenario.participants
            calendar_state = list(scenario.calendar_state)
            pending_requests = list(scenario.pending_requests)
            all_requests = list(scenario.all_requests)

        # Partial observability: hide preferences on HARD
        self._hidden_prefs = {}
        is_hard = scenario.scenario_id.upper().startswith("HARD")
        if is_hard:
            for name, p in participants.items():
                if p.preferred_hours:
                    self._hidden_prefs[name] = list(p.preferred_hours)
                    p.preferred_hours = []

        # Dynamic followups
        self._dynamic_followups = list(scenario.dynamic_followups or [])
        self._followup_trigger_step = max(3, scenario.max_turns // 2)
        self._protected_event_ids, self._recovery_specs = self._configure_operational_mechanics(scenario.scenario_id)

        inv_budget = len(pending_requests) + (3 if is_hard else 1)
        escalation_budget = 2 if is_hard else 1

        self._state = MeetingNegotiatorV1State(
            episode_id=str(uuid4()),
            current_time_utc=scenario.current_time_utc,
            participants=participants,
            calendar_state=calendar_state,
            pending_requests=pending_requests,
            all_requests=all_requests,
            last_action_feedback=f"Scenario loaded: {scenario.scenario_id} - {scenario.description}",
            scenario_id=scenario.scenario_id,
            seed=seed,
            max_turns=scenario.max_turns,
            turn_count=0,
            is_done=False,
            total_reward=0.0,
            score=None,
            hidden_preferences=self._hidden_prefs,
            inspected_participants=[],
            investigation_budget=inv_budget,
            investigation_used=0,
            undo_available=False,
            last_action_snapshot=None,
            total_requests_seen=len(all_requests),
            requests_completed=0,
            system_state="stable",
            escalation_budget_remaining=escalation_budget,
            protected_event_ids=list(self._protected_event_ids),
            modified_existing_event_ids=[],
        )

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: MeetingNegotiatorV1Action) -> MeetingNegotiatorV1Observation:
        if self._state.is_done:
            return self._build_observation(reward=0.0, done=True)

        self._state.turn_count += 1
        reward = -STEP_PENALTY
        reward_components: Dict[str, float] = {"step_cost": -STEP_PENALTY}
        feedback = ""
        done = False

        # Inject dynamic followups mid-episode
        if (self._state.turn_count == self._followup_trigger_step
                and self._dynamic_followups
                and self._state.dynamic_requests_injected == 0):
            for fu in self._dynamic_followups:
                self._state.pending_requests.append(fu)
                if not any(r.request_id == fu.request_id for r in self._state.all_requests):
                    self._state.all_requests.append(fu)
                self._state.total_requests_seen += 1
            self._state.dynamic_requests_injected = len(self._dynamic_followups)
            feedback += f"[QUEUE UPDATE] {len(self._dynamic_followups)} new request(s) arrived. "
            self._state.queue_warning = "New requests arrived mid-session."

        # Max turns check
        if self._state.turn_count >= self._state.max_turns and action.command != "SubmitFinalCalendar":
            self._state.is_done = True
            self._state.score = 0.01
            feedback += "Max turns exceeded without submission. Episode failed."
            reward -= INVALID_ACTION_PENALTY
            reward_components["max_turns_penalty"] = -INVALID_ACTION_PENALTY
            self._state.total_reward += reward
            self._state.last_action_feedback = feedback
            self._state.last_reward_components = reward_components
            return self._build_observation(reward=reward, done=True)

        cmd = action.command
        if cmd == "CheckAvailability":
            reward, feedback, rc = self._handle_check_availability(action, reward)
            reward_components.update(rc)
        elif cmd == "ScheduleNew":
            reward, feedback, rc = self._handle_schedule_new(action, reward)
            reward_components.update(rc)
        elif cmd == "RescheduleExisting":
            reward, feedback, rc = self._handle_reschedule_existing(action, reward)
            reward_components.update(rc)
        elif cmd == "SubmitFinalCalendar":
            reward, feedback, rc, done = self._handle_submit(reward)
            reward_components.update(rc)
        elif cmd == "InspectParticipant":
            reward, feedback, rc = self._handle_inspect_participant(action, reward)
            reward_components.update(rc)
        elif cmd == "ListConflicts":
            reward, feedback, rc = self._handle_list_conflicts(action, reward)
            reward_components.update(rc)
        elif cmd == "GetPolicy":
            feedback = self._get_policy_text()
            reward_components["info_action"] = 0.0
        elif cmd == "UndoLastAction":
            reward, feedback, rc = self._handle_undo(reward)
            reward_components.update(rc)
        else:
            reward -= INVALID_ACTION_PENALTY
            feedback = f"Unknown command: {action.command}."
            reward_components["invalid_action"] = -INVALID_ACTION_PENALTY

        self._state.total_reward += reward
        self._state.last_action_feedback = feedback
        self._state.last_reward_components = reward_components

        return self._build_observation(reward=reward, done=done or self._state.is_done)

    @property
    def state(self) -> MeetingNegotiatorV1State:
        return self._state

    # ── New Action Handlers ────────────────────────────────────────

    def _handle_inspect_participant(
        self, action: MeetingNegotiatorV1Action, reward: float
    ) -> Tuple[float, str, Dict[str, float]]:
        rc: Dict[str, float] = {}
        if not action.target_id:
            rc["invalid_action"] = -INVALID_ACTION_PENALTY
            return reward - INVALID_ACTION_PENALTY, "InspectParticipant requires target_id (participant name).", rc

        name = action.target_id
        participant = self._state.participants.get(name)
        if participant is None:
            rc["invalid_action"] = -INVALID_ACTION_PENALTY
            return reward - INVALID_ACTION_PENALTY, f"Participant '{name}' not found.", rc

        # Use investigation budget
        self._state.investigation_used += 1
        if self._state.investigation_used > self._state.investigation_budget:
            rc["investigation_overbudget"] = -0.02
            reward -= 0.02

        # Reveal hidden preferences if on HARD
        if name in self._hidden_prefs and name not in self._state.inspected_participants:
            participant.preferred_hours = self._hidden_prefs[name]
            self._state.inspected_participants.append(name)

        # Build detailed info
        events_for = [e for e in self._state.calendar_state if name in e.attendees]
        event_strs = [f"  {e.event_id}: {e.start_time_utc} ({e.duration_minutes}min, {e.priority})" for e in events_for]
        events_text = "\n".join(event_strs) if event_strs else "  No scheduled events."

        pref_text = ", ".join(participant.preferred_hours) if participant.preferred_hours else "Not available (hidden or none)"

        info = (
            f"Participant: {name}\n"
            f"  Timezone: {participant.timezone}\n"
            f"  Working Hours: {', '.join(participant.working_hours)}\n"
            f"  Preferred Hours: {pref_text}\n"
            f"  Scheduled Events:\n{events_text}"
        )
        return reward, info, rc

    def _handle_list_conflicts(
        self, action: MeetingNegotiatorV1Action, reward: float
    ) -> Tuple[float, str, Dict[str, float]]:
        rc: Dict[str, float] = {}
        if not action.target_id or not action.proposed_start_utc:
            rc["invalid_action"] = -INVALID_ACTION_PENALTY
            return reward - INVALID_ACTION_PENALTY, "ListConflicts requires target_id and proposed_start_utc.", rc

        request = self._find_request(action.target_id)
        if request is None:
            rc["invalid_action"] = -INVALID_ACTION_PENALTY
            return reward - INVALID_ACTION_PENALTY, f"Request {action.target_id} not found.", rc

        start_dt = self._parse_utc(action.proposed_start_utc)
        end_dt = start_dt + timedelta(minutes=request.duration_minutes)
        conflicts = self._find_conflicts(request.attendees, start_dt, end_dt)

        if not conflicts:
            return reward, "No conflicts found at this time slot.", rc

        lines = []
        for c in conflicts:
            c_start = self._parse_utc(c.start_time_utc)
            c_end = c_start + timedelta(minutes=c.duration_minutes)
            bumpable = PRIORITY_ORDER[c.priority] < PRIORITY_ORDER[request.priority]
            lines.append(
                f"  {c.event_id}: {c.start_time_utc}-{self._format_utc(c_end)} "
                f"priority={c.priority} attendees={c.attendees} "
                f"{'[BUMPABLE]' if bumpable else '[BLOCKING]'}"
            )
        return reward, f"Conflicts at {action.proposed_start_utc}:\n" + "\n".join(lines), rc

    def _handle_undo(self, reward: float) -> Tuple[float, str, Dict[str, float]]:
        rc: Dict[str, float] = {}
        if not self._state.undo_available or not self._state.last_action_snapshot:
            rc["invalid_action"] = -INVALID_ACTION_PENALTY
            return reward - INVALID_ACTION_PENALTY, "No action to undo.", rc

        snap = self._state.last_action_snapshot
        self._state.calendar_state = [ScheduledEvent(**e) for e in snap["calendar_state"]]
        self._state.pending_requests = [MeetingRequest(**r) for r in snap["pending_requests"]]
        self._state.system_state = snap["system_state"]
        self._state.escalation_budget_remaining = snap["escalation_budget_remaining"]
        self._state.triggered_followups = list(snap["triggered_followups"])
        self._state.resolved_recovery_requests = list(snap["resolved_recovery_requests"])
        self._state.pending_recovery_request_ids = list(snap["pending_recovery_request_ids"])
        self._state.modified_existing_event_ids = list(snap["modified_existing_event_ids"])
        self._state.last_transition = snap["last_transition"]
        self._state.queue_warning = snap["queue_warning"]
        self._state.state_transition_log = list(snap["state_transition_log"])
        self._state.undo_available = False
        self._state.last_action_snapshot = None
        rc["undo"] = -STEP_PENALTY
        return reward - STEP_PENALTY, "Last action undone. Calendar and pending requests restored.", rc

    def _get_policy_text(self) -> str:
        return (
            "SCORING POLICY:\n"
            "Final score = base components with stability penalties and recovery credit:\n"
            "  1. Completion (0.35): fraction of requests scheduled\n"
            "  2. Deadline Compliance (0.20): meetings finish before deadline\n"
            "  3. Working Hours (0.15): all within working hours\n"
            "  4. Preference Quality (0.10): preferred hours respected\n"
            "  5. Conflict Avoidance (0.10): no double-bookings\n"
            "  6. Efficiency (0.05): fewer steps = higher score\n"
            "  7. Investigation Discipline (0.05): inspect before acting (HARD only)\n"
            "  8. Stability Penalty: harmful transitions reduce final score\n"
            "  9. Recovery Credit: fixing recovery work restores some score\n\n"
            "PER-STEP PENALTIES:\n"
            f"  Step cost: -{STEP_PENALTY}\n"
            f"  Invalid action: -{INVALID_ACTION_PENALTY}\n"
            f"  Conflict attempt: -{CONFLICT_ACTION_PENALTY}\n"
            f"  Preference violation: -{PREFERENCE_PENALTY_PER_ATTENDEE} per attendee\n\n"
            "BONUSES:\n"
            f"  Schedule success: +{SCHEDULE_SUCCESS_REWARD}\n"
            f"  Reschedule success: +{RESCHEDULE_SUCCESS_REWARD}\n"
            f"  Priority bonus: up to +{PRIORITY_BONUS}\n"
            f"  Early deadline: +{DEADLINE_PROXIMITY_BONUS}"
        )

    def _handle_submit(self, reward: float) -> Tuple[float, str, Dict[str, float], bool]:
        rc: Dict[str, float] = {}
        # Serialize state for grading
        all_reqs = [r.model_dump() for r in self._state.all_requests]
        cal = [e.model_dump() for e in self._state.calendar_state]
        parts = {n: p.model_dump() for n, p in self._state.participants.items()}
        # Restore hidden prefs for scoring
        for name, prefs in self._hidden_prefs.items():
            if name in parts:
                parts[name]["preferred_hours"] = prefs

        score, breakdown = compute_final_score(
            all_requests=all_reqs,
            calendar_state=cal,
            participants=parts,
            turn_count=self._state.turn_count,
            max_turns=self._state.max_turns,
            inspected_participants=self._state.inspected_participants,
            scenario_id=self._state.scenario_id,
            system_state=self._state.system_state,
            escalation_budget_remaining=self._state.escalation_budget_remaining,
            triggered_followups=self._state.triggered_followups,
            resolved_recovery_requests=self._state.resolved_recovery_requests,
            modified_existing_events=[
                event for event in cal
                if event.get("event_id") in self._state.modified_existing_event_ids
                and event.get("request_id") not in {req.get("request_id") for req in all_reqs}
            ],
        )
        self._state.score = score
        self._state.is_done = True
        self._state.reward_breakdown = breakdown
        reward += score
        rc.update(breakdown)
        feedback = f"Final submission received. Score={score:.4f}. Breakdown: {breakdown}"
        return reward, feedback, rc, True

    # ── Original Action Handlers (preserved logic) ─────────────────

    def _save_snapshot(self):
        self._state.last_action_snapshot = {
            "calendar_state": [e.model_dump() for e in self._state.calendar_state],
            "pending_requests": [r.model_dump() for r in self._state.pending_requests],
            "system_state": self._state.system_state,
            "escalation_budget_remaining": self._state.escalation_budget_remaining,
            "triggered_followups": list(self._state.triggered_followups),
            "resolved_recovery_requests": list(self._state.resolved_recovery_requests),
            "pending_recovery_request_ids": list(self._state.pending_recovery_request_ids),
            "modified_existing_event_ids": list(self._state.modified_existing_event_ids),
            "last_transition": self._state.last_transition,
            "queue_warning": self._state.queue_warning,
            "state_transition_log": list(self._state.state_transition_log),
        }
        self._state.undo_available = True

    def _handle_check_availability(
        self, action: MeetingNegotiatorV1Action, reward: float
    ) -> Tuple[float, str, Dict[str, float]]:
        rc: Dict[str, float] = {}
        if not action.target_id or not action.proposed_start_utc:
            rc["invalid_action"] = -INVALID_ACTION_PENALTY
            return reward - INVALID_ACTION_PENALTY, "CheckAvailability requires target_id and proposed_start_utc.", rc

        request = self._find_request(action.target_id)
        if request is None:
            rc["invalid_action"] = -INVALID_ACTION_PENALTY
            return reward - INVALID_ACTION_PENALTY, f"Request {action.target_id} not found.", rc

        start_dt = self._parse_utc(action.proposed_start_utc)
        end_dt = start_dt + timedelta(minutes=request.duration_minutes)
        hard_issues, conflicts = self._evaluate_constraints(request, start_dt, end_dt)
        issues = list(hard_issues)
        if conflicts:
            conflict_names = ", ".join([c.event_id for c in conflicts])
            issues.append(f"conflicts: {conflict_names}")
        if issues:
            return reward, f"Slot NOT available: {', '.join(issues)}.", rc

        # Add preference info
        pref_violations = self._preference_violations(request.attendees, start_dt, end_dt)
        pref_note = ""
        if pref_violations:
            pref_note = f" (Outside preferred hours: {', '.join(pref_violations)})"
        return reward, f"Slot available for all attendees.{pref_note}", rc

    def _handle_schedule_new(
        self, action: MeetingNegotiatorV1Action, reward: float
    ) -> Tuple[float, str, Dict[str, float]]:
        rc: Dict[str, float] = {}
        if not action.target_id or not action.proposed_start_utc:
            rc["invalid_action"] = -INVALID_ACTION_PENALTY
            return reward - INVALID_ACTION_PENALTY, "ScheduleNew requires target_id and proposed_start_utc.", rc

        request = self._find_request(action.target_id)
        if request is None:
            rc["invalid_action"] = -INVALID_ACTION_PENALTY
            return reward - INVALID_ACTION_PENALTY, f"Request {action.target_id} not found in pending requests.", rc

        self._save_snapshot()
        start_dt = self._parse_utc(action.proposed_start_utc)
        end_dt = start_dt + timedelta(minutes=request.duration_minutes)

        hard_issues, conflicts = self._evaluate_constraints(request, start_dt, end_dt)
        if hard_issues:
            reward -= INVALID_ACTION_PENALTY
            rc["constraint_violation"] = -INVALID_ACTION_PENALTY
            return reward, f"Schedule failed: {', '.join(hard_issues)}.", rc

        blocking = self._blocking_conflicts(conflicts, request.priority)
        if blocking:
            reward -= CONFLICT_ACTION_PENALTY
            rc["blocking_conflict"] = -CONFLICT_ACTION_PENALTY
            conflict_ids = ", ".join([c.event_id for c in blocking])
            return reward, f"Schedule failed due to higher/equal priority conflicts: {conflict_ids}.", rc

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
        self._state.requests_completed += 1

        reward += SCHEDULE_SUCCESS_REWARD
        rc["schedule_success"] = SCHEDULE_SUCCESS_REWARD

        priority_level = PRIORITY_ORDER[request.priority]
        p_bonus = PRIORITY_BONUS * (priority_level / 3)
        reward += p_bonus
        rc["priority_bonus"] = p_bonus

        deadline_dt = self._parse_utc(request.deadline_utc)
        minutes_remaining = (deadline_dt - end_dt).total_seconds() / 60
        if minutes_remaining > 30:
            reward += DEADLINE_PROXIMITY_BONUS
            rc["deadline_bonus"] = DEADLINE_PROXIMITY_BONUS

        feedback = f"Scheduled {request.request_id} at {new_event.start_time_utc}."
        pref_violations = self._preference_violations(request.attendees, start_dt, end_dt)
        if pref_violations:
            pv_penalty = PREFERENCE_PENALTY_PER_ATTENDEE * len(pref_violations)
            reward -= pv_penalty
            rc["preference_violation"] = -pv_penalty
            feedback += f" Outside preferred hours: {', '.join(pref_violations)} (-{pv_penalty:.2f})."
        if bumped:
            feedback += f" Bumped lower-priority events: {', '.join(bumped)}."

        harm_reward, harm_feedback, harm_rc = self._apply_post_action_transitions(
            action_kind="schedule",
            target_request=request,
            bumped_ids=bumped,
            start_dt=start_dt,
        )
        reward += harm_reward
        rc.update(harm_rc)
        if harm_feedback:
            feedback += f" {harm_feedback}"

        recovery_reward, recovery_feedback, recovery_rc = self._handle_recovery_resolution(request.request_id)
        reward += recovery_reward
        rc.update(recovery_rc)
        if recovery_feedback:
            feedback += f" {recovery_feedback}"

        return round(reward, 4), feedback, rc

    def _handle_reschedule_existing(
        self, action: MeetingNegotiatorV1Action, reward: float
    ) -> Tuple[float, str, Dict[str, float]]:
        rc: Dict[str, float] = {}
        if not action.target_id or not action.proposed_start_utc:
            rc["invalid_action"] = -INVALID_ACTION_PENALTY
            return reward - INVALID_ACTION_PENALTY, "RescheduleExisting requires target_id and proposed_start_utc.", rc

        event = self._find_event(action.target_id)
        if event is None:
            rc["invalid_action"] = -INVALID_ACTION_PENALTY
            return reward - INVALID_ACTION_PENALTY, f"Event {action.target_id} not found.", rc

        # PATCH: Prevent direct rescheduling of protected structural events
        if event.event_id in self._protected_event_ids:
            reward -= CONFLICT_ACTION_PENALTY
            rc["protected_event_violation"] = -CONFLICT_ACTION_PENALTY
            return reward, f"Reschedule failed: {event.event_id} is a protected event and cannot be moved directly. You must work around it or trigger a cascade.", rc    

        self._save_snapshot()
        start_dt = self._parse_utc(action.proposed_start_utc)
        end_dt = start_dt + timedelta(minutes=event.duration_minutes)

        temp_request = MeetingRequest(
            request_id=event.request_id or f"TEMP-{event.event_id}",
            attendees=event.attendees, duration_minutes=event.duration_minutes,
            priority=event.priority, deadline_utc=self._request_deadline(event.request_id),
            title="Reschedule",
        )

        hard_issues, conflicts = self._evaluate_constraints(temp_request, start_dt, end_dt, ignore_event_id=event.event_id)
        if hard_issues:
            reward -= INVALID_ACTION_PENALTY
            rc["constraint_violation"] = -INVALID_ACTION_PENALTY
            return reward, f"Reschedule failed: {', '.join(hard_issues)}.", rc

        blocking = self._blocking_conflicts(conflicts, event.priority)
        if blocking:
            reward -= CONFLICT_ACTION_PENALTY
            rc["blocking_conflict"] = -CONFLICT_ACTION_PENALTY
            conflict_ids = ", ".join([c.event_id for c in blocking])
            return reward, f"Reschedule failed due to higher/equal priority conflicts: {conflict_ids}.", rc

        bumped = self._bump_conflicts(conflicts, event.priority)
        event.start_time_utc = self._format_utc(start_dt)
        if event.event_id not in self._state.modified_existing_event_ids:
            self._state.modified_existing_event_ids.append(event.event_id)

        reward += RESCHEDULE_SUCCESS_REWARD
        rc["reschedule_success"] = RESCHEDULE_SUCCESS_REWARD

        feedback = f"Rescheduled {event.event_id} to {event.start_time_utc}."
        pref_violations = self._preference_violations(event.attendees, start_dt, end_dt)
        if pref_violations:
            pv_penalty = PREFERENCE_PENALTY_PER_ATTENDEE * len(pref_violations)
            reward -= pv_penalty
            rc["preference_violation"] = -pv_penalty
            feedback += f" Outside preferred hours: {', '.join(pref_violations)} (-{pv_penalty:.2f})."
        if bumped:
            feedback += f" Bumped lower-priority events: {', '.join(bumped)}."

        harm_reward, harm_feedback, harm_rc = self._apply_post_action_transitions(
            action_kind="reschedule",
            target_event=event,
            bumped_ids=bumped,
            start_dt=start_dt,
        )
        reward += harm_reward
        rc.update(harm_rc)
        if harm_feedback:
            feedback += f" {harm_feedback}"

        return round(reward, 4), feedback, rc

    # ── Constraint Evaluation (unchanged logic) ────────────────────

    def _evaluate_constraints(self, request, start_dt, end_dt, ignore_event_id=None):
        hard_issues = []
      
        if start_dt < self._parse_utc(self._state.current_time_utc):
            hard_issues.append("proposed time is in the past")
        if request.deadline_utc and end_dt > self._parse_utc(request.deadline_utc):
            hard_issues.append("misses deadline")
        for attendee in request.attendees:
            participant = self._state.participants.get(attendee)
            if participant is None:
                hard_issues.append(f"unknown attendee {attendee}")
                continue
            try:
                if not self._within_working_hours(participant, start_dt, end_dt):
                    hard_issues.append(f"{attendee} outside working hours")
            except ValueError as exc:
                hard_issues.append(str(exc))
        conflicts = self._find_conflicts(request.attendees, start_dt, end_dt, ignore_event_id)
        return hard_issues, conflicts

    def _find_conflicts(self, attendees, start_dt, end_dt, ignore_event_id=None):
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

    def _blocking_conflicts(self, conflicts, new_priority):
        return [e for e in conflicts if PRIORITY_ORDER[e.priority] >= PRIORITY_ORDER[new_priority]]

    def _bump_conflicts(self, conflicts, new_priority):
        bumped_ids = []
        for event in list(conflicts):
            if PRIORITY_ORDER[event.priority] < PRIORITY_ORDER[new_priority]:
                if event in self._state.calendar_state:
                    self._state.calendar_state.remove(event)
                bumped_ids.append(event.event_id)
                request = self._request_from_event(event)
                if request and not self._request_in_pending(request.request_id):
                    self._state.pending_requests.append(request)
        return bumped_ids

    def _request_from_event(self, event):
        if event.request_id:
            for req in self._state.all_requests:
                if req.request_id == event.request_id:
                    return req
        new_request = MeetingRequest(
            request_id=f"REQ-BUMP-{event.event_id}", attendees=event.attendees,
            duration_minutes=event.duration_minutes, priority=event.priority,
            deadline_utc=self._format_utc(self._parse_utc(self._state.current_time_utc) + timedelta(days=1)),
            title="Bumped meeting",
        )
        if not any(r.request_id == new_request.request_id for r in self._state.all_requests):
            self._state.all_requests.append(new_request)
        return new_request

    def _request_in_pending(self, request_id):
        return any(r.request_id == request_id for r in self._state.pending_requests)

    # ── Time Helpers ───────────────────────────────────────────────

    def _parse_utc(self, value: str) -> datetime:
        return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)

    def _format_utc(self, value: datetime) -> str:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    def _tz_offset(self, tz: str) -> timezone:
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

    def _preference_violations(self, attendees, start_dt, end_dt):
        violations = []
        for attendee in attendees:
            participant = self._state.participants.get(attendee)
            if participant is None:
                continue
            if participant.preferred_hours and not self._within_preferred_hours(participant, start_dt, end_dt):
                violations.append(attendee)
        return violations

    def _within_working_hours(self, participant, start_dt, end_dt):
        tz = self._tz_offset(participant.timezone)
        local_start = start_dt.astimezone(tz)
        local_end = end_dt.astimezone(tz)
        return self._within_blocks(local_start, local_end, participant.working_hours)

    def _within_preferred_hours(self, participant, start_dt, end_dt):
        if not participant.preferred_hours:
            return True
        try:
            tz = self._tz_offset(participant.timezone)
        except ValueError:
            return False
        local_start = start_dt.astimezone(tz)
        local_end = end_dt.astimezone(tz)
        return self._within_blocks(local_start, local_end, participant.preferred_hours)

    def _within_blocks(self, local_start, local_end, blocks):
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

    # ── Utilities ──────────────────────────────────────────────────

    def _find_request(self, request_id):
        for req in self._state.pending_requests:
            if req.request_id == request_id:
                return req
        return None

    def _find_event(self, event_id):
        for event in self._state.calendar_state:
            if event.event_id == event_id:
                return event
        return None

    def _request_deadline(self, request_id):
        if request_id:
            for req in self._state.all_requests:
                if req.request_id == request_id:
                    return req.deadline_utc
        return self._format_utc(self._parse_utc(self._state.current_time_utc) + timedelta(days=1))

    def _configure_operational_mechanics(
        self, scenario_id: str
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        scenario_key = (scenario_id or "").upper()
        protected: List[str] = []
        specs: Dict[str, Dict[str, Any]] = {}

        if scenario_key == "MEDIUM_B":
            protected = ["EVT-DAY2-HOLD"]
            specs["REQ-MEDB-HOLD"] = {
                "followup": MeetingRequest(
                    request_id="REQ-MEDB-RECOVERY",
                    attendees=["Alice", "Bob"],
                    duration_minutes=30,
                    priority="medium",
                    deadline_utc="2026-01-16T17:30Z",
                    title="Manager Apology Sync",
                ),
                "message": "Bumping the placeholder strained the session; a recovery sync was added.",
            }
        elif scenario_key == "MEDIUM_C":
            protected = ["EVT-BOB-LOW"]
            specs["REQ-BOB-LOW"] = {
                "followup": MeetingRequest(
                    request_id="REQ-MEDC-RECOVERY",
                    attendees=["Alice"],
                    duration_minutes=30,
                    priority="medium",
                    deadline_utc="2026-01-15T17:30Z",
                    title="Customer Recovery Update",
                ),
                "message": "Bob's bumped background block represented a customer commitment; recovery work was added.",
            }
        elif scenario_key == "HARD_B":
            protected = ["EVT-CEO-URGENT"]
            specs["EVT-CEO-URGENT"] = {
                "followup": MeetingRequest(
                    request_id="REQ-HARDB-EXEC-RECOVERY",
                    attendees=["CEO", "CTO"],
                    duration_minutes=30,
                    priority="urgent",
                    deadline_utc="2026-01-15T21:00Z",
                    title="Executive Escalation Recovery",
                ),
                "message": "The protected executive block was disturbed; an escalation recovery sync is now required.",
            }
        elif scenario_key == "HARD_C":
            protected = ["EVT-TRAP-BUMP"]
            specs["EVT-TRAP-BUMP"] = {
                "followup": MeetingRequest(
                    request_id="REQ-HARDC-DECISION-RECOVERY",
                    attendees=["CEO", "Alice"],
                    duration_minutes=30,
                    priority="high",
                    deadline_utc="2026-01-15T18:00Z",
                    title="Board Decision Recovery",
                ),
                "message": "Moving the protected trap block disrupted the board-prep chain; recovery work is now required.",
            }

        return protected, specs

    def _apply_post_action_transitions(
        self,
        action_kind: str,
        bumped_ids: List[str],
        start_dt: datetime,
        target_request: Optional[MeetingRequest] = None,
        target_event: Optional[ScheduledEvent] = None,
    ) -> Tuple[float, str, Dict[str, float]]:
        total_reward = 0.0
        rc: Dict[str, float] = {}
        messages: List[str] = []

        for event_id in bumped_ids:
            if event_id in self._protected_event_ids:
                harm_reward, harm_feedback, harm_rc = self._trigger_escalation(event_id)
                total_reward += harm_reward
                rc.update(harm_rc)
                if harm_feedback:
                    messages.append(harm_feedback)
                snapshot_event = None
                if self._state.last_action_snapshot:
                    for saved in self._state.last_action_snapshot["calendar_state"]:
                        if saved["event_id"] == event_id:
                            snapshot_event = saved
                            break
                if snapshot_event and snapshot_event.get("request_id"):
                    harm_reward, harm_feedback, harm_rc = self._trigger_escalation(snapshot_event["request_id"])
                    total_reward += harm_reward
                    rc.update(harm_rc)
                    if harm_feedback:
                        messages.append(harm_feedback)

        if action_kind == "reschedule" and target_event is not None:
            original_start = None
            if self._state.last_action_snapshot:
                for saved in self._state.last_action_snapshot["calendar_state"]:
                    if saved["event_id"] == target_event.event_id:
                        original_start = saved["start_time_utc"]
                        break
            if (
                target_event.event_id in self._protected_event_ids
                and original_start is not None
                and original_start != self._format_utc(start_dt)
            ):
                harm_reward, harm_feedback, harm_rc = self._trigger_escalation(target_event.event_id)
                total_reward += harm_reward
                rc.update(harm_rc)
                if harm_feedback:
                    messages.append(harm_feedback)

        return total_reward, " ".join(messages).strip(), rc

    def _trigger_escalation(self, key: str) -> Tuple[float, str, Dict[str, float]]:
        spec = self._recovery_specs.get(key)
        if spec is None:
            return 0.0, "", {}

        followup: MeetingRequest = spec["followup"]
        if followup.request_id not in self._state.triggered_followups:
            if not self._request_in_pending(followup.request_id):
                self._state.pending_requests.append(followup)
            if not any(r.request_id == followup.request_id for r in self._state.all_requests):
                self._state.all_requests.append(followup)
            self._state.total_requests_seen += 1
            self._state.triggered_followups.append(followup.request_id)
            if followup.request_id not in self._state.pending_recovery_request_ids:
                self._state.pending_recovery_request_ids.append(followup.request_id)
            self._state.escalation_budget_remaining -= 1

        if self._state.escalation_budget_remaining < 0:
            self._state.system_state = "escalated"
        elif self._state.pending_recovery_request_ids:
            self._state.system_state = "recovery_needed"
        else:
            self._state.system_state = "strained"

        self._state.last_transition = f"harm:{key}"
        self._state.state_transition_log.append(self._state.last_transition)
        self._state.queue_warning = spec["message"]

        return -0.05, spec["message"], {"state_transition_penalty": -0.05}

    def _handle_recovery_resolution(self, request_id: str) -> Tuple[float, str, Dict[str, float]]:
        if request_id not in self._state.pending_recovery_request_ids:
            return 0.0, "", {}

        self._state.pending_recovery_request_ids.remove(request_id)
        if request_id not in self._state.resolved_recovery_requests:
            self._state.resolved_recovery_requests.append(request_id)

        self._state.last_transition = f"recovery:{request_id}"
        self._state.state_transition_log.append(self._state.last_transition)
        if self._state.pending_recovery_request_ids:
            self._state.system_state = "strained"
            self._state.queue_warning = "Some recovery work is still pending."
        else:
            self._state.system_state = "stable"
            self._state.queue_warning = None

        return 0.03, f"Recovery request {request_id} resolved; session stability improved.", {"recovery_credit_step": 0.03}

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
            last_reward_components=self._state.last_reward_components,
            reward_breakdown=self._state.reward_breakdown,
            available_commands=list(VALID_COMMANDS),
            investigation_budget_remaining=max(0, self._state.investigation_budget - self._state.investigation_used),
            total_requests_seen=self._state.total_requests_seen,
            requests_completed=self._state.requests_completed,
            system_state=self._state.system_state,
            escalation_budget_remaining=self._state.escalation_budget_remaining,
            protected_event_ids=self._state.protected_event_ids,
            last_transition=self._state.last_transition,
            queue_warning=self._state.queue_warning,
        )

    # ── Scenario Selection ─────────────────────────────────────────

    def _select_scenario(self, reset_count: int, scenario_id: Optional[str] = None) -> ScenarioSpec:
        """Delegate scenario loading to server.scenarios registry."""
        return get_scenario(scenario_id=scenario_id, reset_count=reset_count)
