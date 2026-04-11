"""Tests for per-task grader functions."""

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest
from server.graders import grade_easy, grade_medium, grade_hard


def _make_request(req_id="R1", attendees=None, deadline="2026-01-15T17:00Z"):
    return {"request_id": req_id, "attendees": attendees or ["Alice"], "duration_minutes": 60,
            "priority": "medium", "deadline_utc": deadline, "title": "Test"}


def _make_event(req_id="R1", attendees=None, start="2026-01-15T10:00Z"):
    return {"event_id": f"EVT-{req_id}", "attendees": attendees or ["Alice"],
            "start_time_utc": start, "duration_minutes": 60, "priority": "medium", "request_id": req_id}


def _make_participant(name="Alice"):
    return {"name": name, "timezone": "UTC", "working_hours": ["09:00-17:00"],
            "preferred_hours": ["10:00-12:00"]}


class TestGradeEasy:
    def test_returns_valid_range(self):
        score = grade_easy(
            all_requests=[_make_request()],
            calendar_state=[_make_event()],
            participants={"Alice": _make_participant()},
            turn_count=1, max_turns=9)
        assert 0.01 <= score <= 0.99

    def test_no_data_returns_minimum(self):
        score = grade_easy()
        assert score >= 0.01


class TestGradeHard:
    def test_optimal_score(self):
        score = grade_hard(
            all_requests=[_make_request()],
            calendar_state=[_make_event()],
            participants={"Alice": _make_participant()},
            turn_count=1, max_turns=15,
            inspected_participants=["Alice"])
        assert score >= 0.50

    def test_empty_returns_minimum(self):
        score = grade_hard()
        assert score >= 0.01


class TestGraderDeterminism:
    def test_same_input_same_output(self):
        kwargs = dict(
            all_requests=[_make_request()],
            calendar_state=[_make_event()],
            participants={"Alice": _make_participant()},
            turn_count=3, max_turns=9)
        s1 = grade_easy(**kwargs)
        s2 = grade_easy(**kwargs)
        assert s1 == s2

    def test_medium_deterministic(self):
        kwargs = dict(
            all_requests=[_make_request()],
            calendar_state=[_make_event()],
            participants={"Alice": _make_participant()},
            turn_count=3, max_turns=10)
        s1 = grade_medium(**kwargs)
        s2 = grade_medium(**kwargs)
        assert s1 == s2
