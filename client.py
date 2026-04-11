# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meeting Negotiator V1 Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
try:
    from .models import MeetingNegotiatorV1Action, MeetingNegotiatorV1Observation, MeetingNegotiatorV1State
except ImportError:
    import os as _os
    import sys as _sys
    _pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
    if _pkg_dir not in _sys.path:
        _sys.path.insert(0, _pkg_dir)
    from models import MeetingNegotiatorV1Action, MeetingNegotiatorV1Observation, MeetingNegotiatorV1State

class MeetingNegotiatorV1Env(
    EnvClient[MeetingNegotiatorV1Action, MeetingNegotiatorV1Observation, MeetingNegotiatorV1State]
):
    """
    Client for the Meeting Negotiator V1 Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """

    def _step_payload(self, action: MeetingNegotiatorV1Action) -> Dict:
        return {
            "command": action.command,
            "target_id": action.target_id,
            "proposed_start_utc": action.proposed_start_utc,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MeetingNegotiatorV1Observation]:
        obs_data = payload.get("observation", {})
        observation = MeetingNegotiatorV1Observation(
            current_time_utc=obs_data.get("current_time_utc", ""),
            participants=obs_data.get("participants", {}),
            calendar_state=obs_data.get("calendar_state", []),
            pending_requests=obs_data.get("pending_requests", []),
            last_action_feedback=obs_data.get("last_action_feedback", ""),
            turn_count=obs_data.get("turn_count", 0),
            max_turns=obs_data.get("max_turns", 0),
            score=obs_data.get("score"),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            done=obs_data.get("done", payload.get("done", False)),
            last_reward_components=obs_data.get("last_reward_components", {}),
            reward_breakdown=obs_data.get("reward_breakdown"),
            available_commands=obs_data.get("available_commands", []),
            investigation_budget_remaining=obs_data.get("investigation_budget_remaining", 0),
            total_requests_seen=obs_data.get("total_requests_seen", 0),
            requests_completed=obs_data.get("requests_completed", 0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> MeetingNegotiatorV1State:
        return MeetingNegotiatorV1State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_time_utc=payload.get("current_time_utc", ""),
            scenario_id=payload.get("scenario_id", ""),
            seed=payload.get("seed"),
            max_turns=payload.get("max_turns", 15),
            turn_count=payload.get("turn_count", 0),
            is_done=payload.get("is_done", False),
            total_reward=payload.get("total_reward", 0.0),
            score=payload.get("score"),
            investigation_budget=payload.get("investigation_budget", 0),
            investigation_used=payload.get("investigation_used", 0),
            inspected_participants=payload.get("inspected_participants", []),
            total_requests_seen=payload.get("total_requests_seen", 0),
            requests_completed=payload.get("requests_completed", 0),
        )
