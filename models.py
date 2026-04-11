# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Meeting Negotiator V1 Environment.

Typed Pydantic models for the full OpenEnv spec:
  - MeetingNegotiatorV1Action  (agent → env)
  - MeetingNegotiatorV1Observation  (env → agent)
  - MeetingNegotiatorV1State  (internal env state)
"""

from typing import Dict, List, Optional, Literal, Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# ── Domain Models ──────────────────────────────────────────────────────

class Participant(BaseModel):
    name: str = Field(..., description="Name of the participant")
    timezone: str = Field(..., description="Timezone of the participant (e.g., PST, EST, GMT, UTC, IST)")
    working_hours: List[str] = Field(
        ...,
        description=(
            "List of available working blocks in 'HH:MM-HH:MM' 24-hour format. "
            "E.g., ['09:00-12:00', '13:00-17:00'] to account for lunch breaks."
        ),
    )
    preferred_hours: List[str] = Field(
        default_factory=list,
        description=(
            "Soft preferences for meeting blocks in local time. These are optional and used for satisfaction scoring. "
            "On HARD difficulty, these are hidden until the agent uses InspectParticipant."
        ),
    )


class ScheduledEvent(BaseModel):
    event_id: str = Field(..., description="The unique identifier for this scheduled meeting on the calendar.")
    attendees: List[str] = Field(..., description="List of participant names that are attending this meeting.")
    start_time_utc: str = Field(
        ..., description="The confirmed start time of the meeting in UTC (e.g. '2026-01-15T14:00Z')"
    )
    duration_minutes: int = Field(..., gt=10, description="The length of the meeting in minutes.")
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        ..., description="The priority level of the meeting. Higher priority can bump lower ones if necessary."
    )
    request_id: Optional[str] = Field(
        default=None, description="Link back to the originating MeetingRequest when applicable."
    )


class MeetingRequest(BaseModel):
    request_id: str = Field(..., description="The unique identifier for this pending meeting request.")
    attendees: List[str] = Field(..., description="List of participant names who MUST attend this meeting.")
    duration_minutes: int = Field(..., ge=10, description="The required length of the meeting in minutes.")
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        ..., description="The priority level of the requested meeting."
    )
    deadline_utc: str = Field(
        ...,
        description="The meeting MUST finish by this UTC time.",
    )
    title: str = Field(default="", description="Short human-readable title for the request.")


# ── OpenEnv Types ──────────────────────────────────────────────────────

VALID_COMMANDS = [
    "CheckAvailability",
    "ScheduleNew",
    "RescheduleExisting",
    "SubmitFinalCalendar",
    "InspectParticipant",
    "ListConflicts",
    "GetPolicy",
    "UndoLastAction",
]


class MeetingNegotiatorV1Action(Action):
    """Action space for the Meeting Negotiator V1 environment.

    Expanded action set:
      - CheckAvailability: probe conflicts without penalty
      - ScheduleNew: lock a pending request into a time slot
      - RescheduleExisting: move an existing calendar event
      - SubmitFinalCalendar: finalize and receive the score
      - InspectParticipant: reveal hidden preferences (costs investigation budget)
      - ListConflicts: get a breakdown of conflicts at a proposed time
      - GetPolicy: see the current scoring rules
      - UndoLastAction: reverse the last ScheduleNew or RescheduleExisting
    """

    command: Literal[
        "CheckAvailability",
        "ScheduleNew",
        "RescheduleExisting",
        "SubmitFinalCalendar",
        "InspectParticipant",
        "ListConflicts",
        "GetPolicy",
        "UndoLastAction",
    ] = Field(
        ..., description="Action to execute."
    )
    target_id: Optional[str] = Field(
        None,
        description=(
            "Required for ScheduleNew (use request_id), RescheduleExisting (use event_id), "
            "InspectParticipant (use participant name). Leave null for SubmitFinalCalendar, GetPolicy, UndoLastAction."
        ),
    )
    proposed_start_utc: Optional[str] = Field(
        None,
        description=(
            "The proposed UTC start time (e.g., '2026-01-15T15:00Z'). "
            "Required for CheckAvailability, ScheduleNew, RescheduleExisting, and ListConflicts."
        ),
    )


class MeetingNegotiatorV1Observation(Observation):
    """Observation space for the Meeting Negotiator V1 environment.

    Includes rich state payload, decomposed reward telemetry, and
    partial observability markers.
    """

    current_time_utc: str = Field(
        ..., description="The current simulated time in UTC. Do not schedule meetings in the past."
    )
    participants: Dict[str, Participant] = Field(
        ..., description="Dictionary mapping participant names to their timezone and working hours."
    )
    calendar_state: List[ScheduledEvent] = Field(
        default_factory=list, description="List of all meetings currently locked into the calendar."
    )
    pending_requests: List[MeetingRequest] = Field(
        default_factory=list, description="List of meetings you must successfully schedule to complete the task."
    )
    last_action_feedback: str = Field(
        default="System initialized.",
        description="Crucial feedback from the environment regarding your previous action.",
    )
    turn_count: int = Field(default=0, description="How many actions you have taken so far.")
    max_turns: int = Field(
        default=15,
        description="The maximum allowed turns. You MUST use 'SubmitFinalCalendar' before reaching this limit, or you fail.",
    )
    score: Optional[float] = Field(
        default=None,
        description="Final score in [0.01, 0.99]. Only populated after SubmitFinalCalendar or failure.",
    )
    reward: float = Field(default=0.0, description="Reward for the last action (penalty-based).")
    done: bool = Field(default=False, description="Whether the episode is complete.")

    # ── Reward Transparency ──
    last_reward_components: Dict[str, float] = Field(
        default_factory=dict,
        description="Decomposed reward breakdown for the last action (e.g., step_cost, schedule_success, preference_violation).",
    )
    reward_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description="Full decomposed final score breakdown. Only populated on SubmitFinalCalendar.",
    )

    # ── Action Space Telemetry ──
    available_commands: List[str] = Field(
        default_factory=lambda: list(VALID_COMMANDS),
        description="Commands the agent can legally use in this state.",
    )
    investigation_budget_remaining: int = Field(
        default=0,
        description="Remaining free investigation actions. Extra investigations cost reward.",
    )

    # ── Queue Telemetry ──
    total_requests_seen: int = Field(default=0, description="Total requests seen so far (including dynamic ones).")
    requests_completed: int = Field(default=0, description="Requests successfully scheduled.")

    # ── Stateful Ops Telemetry ──
    system_state: str = Field(default="stable", description="Operational session state: stable, strained, escalated, or recovery_needed.")
    escalation_budget_remaining: int = Field(default=0, description="Remaining budget before the scheduling session escalates.")
    protected_event_ids: List[str] = Field(default_factory=list, description="Existing events that should not be moved or displaced casually.")
    last_transition: Optional[str] = Field(default=None, description="Last state transition summary, if any.")
    queue_warning: Optional[str] = Field(default=None, description="Operational warning surfaced after a harmful transition.")


class MeetingNegotiatorV1State(State):
    """
    Internal state of the environment.
    Returned by env.state() for tracking and debugging by the OpenEnv grader.
    """

    current_time_utc: str = ""
    participants: Dict[str, Participant] = {}
    calendar_state: List[ScheduledEvent] = []
    pending_requests: List[MeetingRequest] = []
    all_requests: List[MeetingRequest] = []
    last_action_feedback: str = ""

    scenario_id: str = ""
    seed: Optional[int] = Field(default=None)
    max_turns: int = Field(default=15)
    turn_count: int = Field(default=0)
    is_done: bool = Field(default=False)
    total_reward: float = Field(default=0.0)
    score: Optional[float] = Field(default=None)

    # ── Partial Observability ──
    hidden_preferences: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Preferences hidden from the agent until InspectParticipant is called.",
    )
    inspected_participants: List[str] = Field(
        default_factory=list,
        description="Participants whose preferences have been revealed.",
    )
    investigation_budget: int = Field(default=0)
    investigation_used: int = Field(default=0)

    # ── Undo Tracking ──
    undo_available: bool = Field(default=False)
    last_action_snapshot: Optional[Dict[str, Any]] = Field(default=None)

    # ── Queue Dynamics ──
    dynamic_requests_injected: int = Field(default=0)
    total_requests_seen: int = Field(default=0)
    requests_completed: int = Field(default=0)

    # ── Stateful Ops Mechanics ──
    system_state: str = Field(default="stable")
    escalation_budget_remaining: int = Field(default=0)
    protected_event_ids: List[str] = Field(default_factory=list)
    state_transition_log: List[str] = Field(default_factory=list)
    last_transition: Optional[str] = Field(default=None)
    queue_warning: Optional[str] = Field(default=None)
    triggered_followups: List[str] = Field(default_factory=list)
    resolved_recovery_requests: List[str] = Field(default_factory=list)
    pending_recovery_request_ids: List[str] = Field(default_factory=list)

    # ── Reward Components ──
    last_reward_components: Dict[str, float] = Field(default_factory=dict)
    reward_breakdown: Optional[Dict[str, float]] = Field(default=None)
