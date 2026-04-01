# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Meeting Negotiator V1 Environment.

The meeting_negotiator_v1 environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, BaseModel
from typing import List, Dict, Optional, Literal

class Participant(BaseModel):
    name : str = Field(..., description="Name of the participant")
    timezone : str = Field(..., description="Timezone of the participant")
    working_hours :List[str] = Field(
        ..., 
        description="List of available working blocks in 'HH:MM-HH:MM' 24-hour format. E.g., ['09:00-12:00', '13:00-17:00'] to account for lunch breaks."
    )

class ScheduledEvent(BaseModel):
    event_id : str = Field(
        ...,
        description="The unique identfier for this scheduled meeting on the calendar."
    )
    attendees : List[str] = Field(
        ...,
        description="List of participant names that are attending this meeting."
    )

    start_time_utc : str = Field(
        ...,
        description="The confirmed start time of the meeting in UTC (e.g. '2026-04-02T14:00Z')"
    )
    duration_minutes : int = Field(
        ...,
        gt=10,
        description="The length of the meeting in minutes."
    )
    priority : Literal["low", "medium", "high", "urgent"] = Field(
        ..., 
        description="The priority level of the meeting. The higher priority meetings can bump the lower ones if necessary."
    )

class MeetingRequest(BaseModel):
    request_id : str = Field(
        ...,
        description="The unique identfier for this pending meeting request."
    )
    attendees: List[str] = Field(
        ..., 
        description="List of participant names who MUST attend this meeting."
    )
    duration_minutes: int = Field(
        ..., 
        ge=10,
        description="The required length of the meeting in minutes."
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        ..., 
        description="The priority level of the requested meeting."
    )
    deadline_utc: str = Field(
        ..., 
        description="The meeting MUST be scheduled to start before this UTC time."
    )


class MeetingNegotiatorV1Action(Action):
    """Action space for the Meeting Negotiator V1 environment."""

    command : Literal["CheckAvailability", "ScheduleNew", "RescheduleExisting", "SubmitFinalCalendar"] = Field(
        ...,
        description="The action to execute. Use CheckAvailability to probe for conflicts without penalty before committing."
    )
    target_id: Optional[str] = Field(
        None, 
        description="Required for ScheduleNew (use request_id) and RescheduleExisting (use event_id). Leave null for SubmitFinalCalendar."
    )
    proposed_start_utc: Optional[str] = Field(
        None, 
        description="The proposed UTC start time (e.g., '2026-04-02T15:00Z'). Required for CheckAvailability, ScheduleNew, and RescheduleExisting."
    )
    # """Action for the Meeting Negotiator V1 environment - just a message to echo."""

    # message: str = Field(..., description="Message to echo back")


class MeetingNegotiatorV1Observation(Observation):
    """Observation space for the Meeting Negotiator V1 environment."""

    current_time_utc : str = Field(
        ...,
        description="The current simulated time in UTC. Do not schedule meetings in the past."
    )
    participants : Dict[str,Participant] = Field(
        ...,
        description="Dictionary mapping participant names to their timezone and working hours."
    )
    calendar_state : List[ScheduledEvent] = Field(
        default_factory=list,
        description="List of all meetings currently locked into the calendar."
    )
    pending_requests : List[MeetingRequest] = Field(
        default_factory=list, 
        description="List of meetings you must successfully schedule to complete the task."
    )
    last_action_feedback: str = Field(
        default="System initialized.", 
        description="Crucial feedback from the environment regarding your previous action (e.g., 'Slot available' or 'Conflict: Bob is busy')."
    )
    turn_count: int = Field(
        default=0, 
        description="How many actions you have taken so far."
    )
    max_turns: int = Field(
        default=15, 
        description="The maximum allowed turns. You MUST use 'SubmitFinalCalendar' before reaching this limit, or you fail."
    )

    # """Observation from the Meeting Negotiator V1 environment - the echoed message."""

    # echoed_message: str = Field(default="", description="The echoed message")
    # message_length: int = Field(default=0, description="Length of the echoed message")


# INTERNAL STATE TRACKING

class MeetingNegotiatorV1State(State):
    """
    The internal state of the environment. 
    This is returned by the env.state() method for tracking and debugging by the OpenEnv grader.
    """
    current_time_utc :  str = ""
    participants: Dict[str, Participant] = {}
    calendar_state: List[ScheduledEvent] = []
    pending_requests: List[MeetingRequest] = []
    last_action_feedback: str = ""

    max_turns: int = Field(default=15)
    is_done: bool = Field(default=False)
    total_reward: float = Field(default=0.0)