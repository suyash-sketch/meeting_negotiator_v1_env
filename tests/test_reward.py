"""Tests for the decomposed reward module."""

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest
from server.reward import compute_final_score


def _make_request(req_id="R1", attendees=None, deadline="2026-01-15T17:00Z", priority="medium"):
    return {"request_id": req_id, "attendees": attendees or ["Alice"], "duration_minutes": 60,
            "priority": priority, "deadline_utc": deadline, "title": "Test"}


def _make_event(req_id="R1", attendees=None, start="2026-01-15T10:00Z", priority="medium"):
    return {"event_id": f"EVT-{req_id}", "attendees": attendees or ["Alice"],
            "start_time_utc": start, "duration_minutes": 60, "priority": priority, "request_id": req_id}


def _make_participant(name="Alice", tz="UTC", wh=None, ph=None):
    return {"name": name, "timezone": tz,
            "working_hours": wh or ["09:00-17:00"],
            "preferred_hours": ph or ["10:00-12:00"]}


class TestPerfectEasyScore:
    def test_single_request_perfect(self):
        score, breakdown = compute_final_score(
            all_requests=[_make_request()],
            calendar_state=[_make_event()],
            participants={"Alice": _make_participant()},
            turn_count=1, max_turns=9,
        )
        assert score >= 0.80
        assert breakdown["completion"] > 0


class TestUnscheduledPenalty:
    def test_missing_request_lowers_completion(self):
        score, breakdown = compute_final_score(
            all_requests=[_make_request(), _make_request(req_id="R2")],
            calendar_state=[_make_event()],
            participants={"Alice": _make_participant()},
            turn_count=1, max_turns=9,
        )
        assert breakdown["completion"] < 0.35
        assert score < 0.90


class TestDeadlineViolation:
    def test_past_deadline_penalizes(self):
        score, breakdown = compute_final_score(
            all_requests=[_make_request(deadline="2026-01-15T10:00Z")],
            calendar_state=[_make_event(start="2026-01-15T10:00Z")],
            participants={"Alice": _make_participant()},
            turn_count=1, max_turns=9,
        )
        assert breakdown["deadline_compliance"] < 0.20


class TestPreferencePenalty:
    def test_outside_preferred_hours(self):
        score, breakdown = compute_final_score(
            all_requests=[_make_request()],
            calendar_state=[_make_event(start="2026-01-15T14:00Z")],
            participants={"Alice": _make_participant(ph=["10:00-12:00"])},
            turn_count=1, max_turns=9,
        )
        assert breakdown["preference_quality"] < 0.10


class TestConflictsNotScored:
    """Terminal calendar from real env cannot double-book; scorer ignores overlap."""

    def test_double_booking_has_no_conflict_component(self):
        _, breakdown = compute_final_score(
            all_requests=[_make_request(), _make_request(req_id="R2")],
            calendar_state=[
                _make_event(start="2026-01-15T10:00Z"),
                _make_event(req_id="R2", start="2026-01-15T10:00Z"),
            ],
            participants={"Alice": _make_participant()},
            turn_count=1, max_turns=9,
        )
        assert "conflict_avoidance" not in breakdown
        assert breakdown["completion"] >= 0.39


class TestComponentsSum:
    def test_all_components_present(self):
        score, breakdown = compute_final_score(
            all_requests=[_make_request()],
            calendar_state=[_make_event()],
            participants={"Alice": _make_participant()},
            turn_count=1, max_turns=9,
        )
        expected_keys = {
            "completion",
            "deadline_compliance",
            "preference_quality",
            "efficiency",
            "investigation_discipline",
            "stability_penalty",
            "recovery_credit",
        }
        assert set(breakdown.keys()) == expected_keys
        assert abs(sum(breakdown.values()) - score) < 0.01 or score in (0.01, 0.99)


class TestScoreRange:
    def test_always_in_range(self):
        score, _ = compute_final_score([], [], {}, 0, 9)
        assert 0.01 <= score <= 0.99
