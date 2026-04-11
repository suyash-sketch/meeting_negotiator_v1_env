"""Tests for seed-based scenario randomization."""

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest
from server.scenario_resolver import resolve_scenario


def _sample_scenario():
    return {
        "participants": {
            "Alice": {"name": "Alice", "timezone": "UTC", "working_hours": ["09:00-17:00"], "preferred_hours": ["10:00-12:00"]},
            "Bob": {"name": "Bob", "timezone": "EST", "working_hours": ["09:00-17:00"], "preferred_hours": ["14:00-16:00"]},
        },
        "calendar_state": [
            {"event_id": "EVT-1", "attendees": ["Alice", "Bob"], "start_time_utc": "2026-01-15T10:00Z",
             "duration_minutes": 60, "priority": "medium"}
        ],
        "pending_requests": [
            {"request_id": "R1", "attendees": ["Alice", "Bob"], "duration_minutes": 60,
             "priority": "medium", "deadline_utc": "2026-01-15T17:00Z", "title": "1:1 Alice/Bob"}
        ],
        "all_requests": [
            {"request_id": "R1", "attendees": ["Alice", "Bob"], "duration_minutes": 60,
             "priority": "medium", "deadline_utc": "2026-01-15T17:00Z", "title": "1:1 Alice/Bob"}
        ],
    }


class TestSameSeed:
    def test_produces_same_scenario(self):
        s1 = resolve_scenario(_sample_scenario(), seed=42)
        s2 = resolve_scenario(_sample_scenario(), seed=42)
        assert s1 == s2


class TestDifferentSeeds:
    def test_produce_different_names(self):
        s1 = resolve_scenario(_sample_scenario(), seed=1)
        s2 = resolve_scenario(_sample_scenario(), seed=999)
        # At least one name or deadline should differ
        names1 = set(s1["participants"].keys())
        names2 = set(s2["participants"].keys())
        deadlines1 = [r["deadline_utc"] for r in s1["all_requests"]]
        deadlines2 = [r["deadline_utc"] for r in s2["all_requests"]]
        assert names1 != names2 or deadlines1 != deadlines2


class TestSeedNone:
    def test_returns_original(self):
        orig = _sample_scenario()
        result = resolve_scenario(orig, seed=None)
        assert set(result["participants"].keys()) == set(orig["participants"].keys())


class TestStructuralInvariant:
    def test_event_count_unchanged(self):
        orig = _sample_scenario()
        resolved = resolve_scenario(orig, seed=42)
        assert len(resolved["calendar_state"]) == len(orig["calendar_state"])
        assert len(resolved["pending_requests"]) == len(orig["pending_requests"])
        assert len(resolved["all_requests"]) == len(orig["all_requests"])

    def test_durations_unchanged(self):
        orig = _sample_scenario()
        resolved = resolve_scenario(orig, seed=42)
        for o, r in zip(orig["calendar_state"], resolved["calendar_state"]):
            assert o["duration_minutes"] == r["duration_minutes"]

    def test_priorities_unchanged(self):
        orig = _sample_scenario()
        resolved = resolve_scenario(orig, seed=42)
        for o, r in zip(orig["calendar_state"], resolved["calendar_state"]):
            assert o["priority"] == r["priority"]
