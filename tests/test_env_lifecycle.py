"""Tests for environment lifecycle: reset, step, done, state."""

import os
import sys

# Ensure project root is on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest
from server.meeting_negotiator_v1_environment import MeetingNegotiatorV1Environment
from models import MeetingNegotiatorV1Action


@pytest.fixture
def env():
    return MeetingNegotiatorV1Environment()


class TestResetReturnsValidObservation:
    def test_easy_reset(self, env):
        obs = env.reset(scenario_id="EASY")
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.turn_count == 0
        assert len(obs.pending_requests) > 0
        assert obs.current_time_utc != ""
        assert obs.score is None

    def test_medium_reset(self, env):
        obs = env.reset(scenario_id="MEDIUM")
        assert obs.done is False
        assert len(obs.participants) >= 3

    def test_hard_reset(self, env):
        obs = env.reset(scenario_id="HARD")
        assert obs.done is False
        assert len(obs.calendar_state) > 0


class TestStepAfterDone:
    def test_step_after_submit_returns_done(self, env):
        env.reset(scenario_id="EASY")
        obs = env.step(MeetingNegotiatorV1Action(command="SubmitFinalCalendar"))
        assert obs.done is True
        obs2 = env.step(MeetingNegotiatorV1Action(command="CheckAvailability", target_id="REQ-EASY-1", proposed_start_utc="2026-01-15T10:00Z"))
        assert obs2.done is True


class TestMaxTurnsForFailure:
    def test_max_turns_exceeded(self, env):
        env.reset(scenario_id="EASY")
        for _ in range(20):
            obs = env.step(MeetingNegotiatorV1Action(command="GetPolicy"))
            if obs.done:
                break
        assert obs.done is True
        assert obs.score is not None
        assert obs.score <= 0.02


class TestSubmitReturnsScore:
    def test_submit_produces_score(self, env):
        env.reset(scenario_id="EASY")
        obs = env.step(MeetingNegotiatorV1Action(command="SubmitFinalCalendar"))
        assert obs.score is not None
        assert 0.0 < obs.score <= 1.0
        assert obs.reward_breakdown is not None
        assert "completion" in obs.reward_breakdown


class TestResetClearsState:
    def test_consecutive_resets(self, env):
        env.reset(scenario_id="EASY")
        env.step(MeetingNegotiatorV1Action(command="SubmitFinalCalendar"))
        obs = env.reset(scenario_id="MEDIUM")
        assert obs.done is False
        assert obs.turn_count == 0
        assert obs.score is None


class TestNewActions:
    def test_inspect_participant(self, env):
        env.reset(scenario_id="EASY")
        obs = env.step(MeetingNegotiatorV1Action(command="InspectParticipant", target_id="Alice"))
        assert "Alice" in obs.last_action_feedback
        assert "Timezone" in obs.last_action_feedback

    def test_list_conflicts(self, env):
        env.reset(scenario_id="HARD")
        obs = env.step(MeetingNegotiatorV1Action(
            command="ListConflicts", target_id="REQ-URGENT-ALL-HANDS",
            proposed_start_utc="2026-01-15T14:00Z"))
        assert obs.done is False

    def test_get_policy(self, env):
        env.reset(scenario_id="EASY")
        obs = env.step(MeetingNegotiatorV1Action(command="GetPolicy"))
        assert "SCORING POLICY" in obs.last_action_feedback

    def test_undo_without_prior(self, env):
        env.reset(scenario_id="EASY")
        obs = env.step(MeetingNegotiatorV1Action(command="UndoLastAction"))
        assert "No action to undo" in obs.last_action_feedback


class TestPartialObservability:
    def test_hard_hides_preferences(self, env):
        obs = env.reset(scenario_id="HARD")
        for name, p in obs.participants.items():
            assert p.preferred_hours == [], f"{name} should have hidden preferences on HARD"

    def test_inspect_reveals_preferences(self, env):
        env.reset(scenario_id="HARD")
        obs = env.step(MeetingNegotiatorV1Action(command="InspectParticipant", target_id="CEO"))
        ceo = obs.participants.get("CEO")
        assert ceo is not None
        assert len(ceo.preferred_hours) > 0


class TestRewardDecomposition:
    def test_step_has_reward_components(self, env):
        env.reset(scenario_id="EASY")
        obs = env.step(MeetingNegotiatorV1Action(command="GetPolicy"))
        assert "step_cost" in obs.last_reward_components

    def test_easy_after_hours_still_rejected(self, env):
        env.reset(scenario_id="EASY")
        obs = env.step(MeetingNegotiatorV1Action(
            command="ScheduleNew",
            target_id="REQ-EASY-1",
            proposed_start_utc="2026-01-15T18:00Z",
        ))
        assert "outside working hours" in obs.last_action_feedback

    def test_hard_after_hours_rejected_after_cleanup(self, env):
        env.reset(scenario_id="HARD_B")
        obs = env.step(MeetingNegotiatorV1Action(
            command="ScheduleNew",
            target_id="REQ-HARDB-DECOY",
            proposed_start_utc="2026-01-15T16:00Z",
        ))
        pending_ids = {req.request_id for req in obs.pending_requests}
        assert "outside working hours" in obs.last_action_feedback
        assert "REQ-HARDB-DECOY" in pending_ids

    def test_modified_existing_event_counts_in_final_compliance(self, env):
        env.reset(scenario_id="HARD")
        env.step(MeetingNegotiatorV1Action(
            command="RescheduleExisting",
            target_id="EVT-ALICE-BOB-URGENT",
            proposed_start_utc="2026-01-15T18:00Z",
        ))
        obs = env.step(MeetingNegotiatorV1Action(command="SubmitFinalCalendar"))
        assert obs.reward_breakdown is not None
        assert obs.reward_breakdown["working_hours_compliance"] < 0.15
        assert obs.reward_breakdown["preference_quality"] < 0.10


class TestDynamicFollowups:
    def test_medium_injects_followups(self, env):
        obs = env.reset(scenario_id="MEDIUM")
        initial_pending = len(obs.pending_requests)
        # Step until followup triggers
        for i in range(15):
            obs = env.step(MeetingNegotiatorV1Action(command="GetPolicy"))
            if obs.done or "QUEUE UPDATE" in obs.last_action_feedback:
                break
        # Either followup was injected or episode ended
        assert obs.done or obs.total_requests_seen >= initial_pending


class TestOperationalTransitions:
    def test_medium_b_protected_bump_triggers_recovery(self, env):
        env.reset(scenario_id="MEDIUM_B")
        obs = env.step(MeetingNegotiatorV1Action(
            command="ScheduleNew",
            target_id="REQ-MEDB-1",
            proposed_start_utc="2026-01-16T14:00Z",
        ))
        pending_ids = {req.request_id for req in obs.pending_requests}
        assert obs.system_state == "recovery_needed"
        assert "REQ-MEDB-RECOVERY" in pending_ids
        assert obs.escalation_budget_remaining == 0

    def test_recovery_request_returns_session_to_stable(self, env):
        env.reset(scenario_id="MEDIUM_B")
        env.step(MeetingNegotiatorV1Action(
            command="ScheduleNew",
            target_id="REQ-MEDB-1",
            proposed_start_utc="2026-01-16T14:00Z",
        ))
        obs = env.step(MeetingNegotiatorV1Action(
            command="ScheduleNew",
            target_id="REQ-MEDB-RECOVERY",
            proposed_start_utc="2026-01-16T15:00Z",
        ))
        pending_ids = {req.request_id for req in obs.pending_requests}
        assert obs.system_state == "stable"
        assert "REQ-MEDB-RECOVERY" not in pending_ids

    def test_hard_c_protected_bump_triggers_recovery(self, env):
        env.reset(scenario_id="HARD_C")
        obs = env.step(MeetingNegotiatorV1Action(
            command="ScheduleNew",
            target_id="REQ-URGENT-DAY1",
            proposed_start_utc="2026-01-15T15:00Z",
        ))
        pending_ids = {req.request_id for req in obs.pending_requests}
        assert "REQ-HARDC-DECISION-RECOVERY" in pending_ids
        assert obs.system_state == "recovery_needed"
