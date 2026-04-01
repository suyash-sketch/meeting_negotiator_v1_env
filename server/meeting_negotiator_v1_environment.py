# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Meeting Negotiator V1 Environment Implementation.
A multi-step constraint satisfaction RL environment for calendar coordination.
"""
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State


try:
    from ..models import MeetingNegotiatorV1Action, MeetingNegotiatorV1Observation, MeetingNegotiatorV1State, Participant, MeetingRequest, ScheduledEvent
except ImportError:
    from models import MeetingNegotiatorV1Action, MeetingNegotiatorV1Observation, MeetingNegotiatorV1State, Participant, MeetingRequest, ScheduledEvent


class MeetingNegotiatorV1Environment(Environment):
    # """
    # A simple echo environment that echoes back messages.

    # This environment is designed for testing the HTTP server infrastructure.
    # It maintains minimal state and simply echoes back whatever message it receives.

    # Example:
    #     >>> env = MeetingNegotiatorV1Environment()
    #     >>> obs = env.reset()
    #     >>> print(obs.echoed_message)  # "Meeting Negotiator V1 environment ready!"
    #     >>>
    #     >>> obs = env.step(MeetingNegotiatorV1Action(message="Hello"))
    #     >>> print(obs.echoed_message)  # "Hello"
    #     >>> print(obs.message_length)  # 5
    # """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).


    """
    The Meeting Negotiator Environment.
    The agent must schedule pending meeting requests into a calendar without 
    creating conflicts and respecting participant working hours.
    """

    
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the environment with an empty state."""
        self._state = MeetingNegotiatorV1State()
        self._reset_count = 0

    def load_scenario(self):
        """
        This is where you will load your Easy/Medium/Hard JSON task configs.
        For now, we will hardcode a basic 'Clean Slate' initialization so you can test it.
        """
        self.state.current_time_utc = "2026-04-01T08:00Z"
        self.state.participants = {
            "Alice" : Participant
        }

    def reset(self) -> MeetingNegotiatorV1Observation:
        """
        Reset the environment.

        Returns:
            MeetingNegotiatorV1Observation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return MeetingNegotiatorV1Observation(
            echoed_message="Meeting Negotiator V1 environment ready!",
            message_length=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: MeetingNegotiatorV1Action) -> MeetingNegotiatorV1Observation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: MeetingNegotiatorV1Action containing the message to echo

        Returns:
            MeetingNegotiatorV1Observation with the echoed message and its length
        """
        self._state.step_count += 1

        message = action.message
        length = len(message)

        # Simple reward: longer messages get higher rewards
        reward = length * 0.1

        return MeetingNegotiatorV1Observation(
            echoed_message=message,
            message_length=length,
            done=False,
            reward=reward,
            metadata={"original_message": message, "step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
