"""Tests for timezone conversion and working hour logic."""

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest
from datetime import datetime, timedelta, timezone
from server.reward import _tz_offset, _within_blocks, _parse_utc


class TestTzOffset:
    def test_est(self):
        tz = _tz_offset("EST")
        dt = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc).astimezone(tz)
        assert dt.hour == 7  # 12 UTC = 7 EST

    def test_ist_half_hour(self):
        tz = _tz_offset("IST")
        dt = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc).astimezone(tz)
        assert dt.hour == 17
        assert dt.minute == 30  # 12:00 UTC = 17:30 IST

    def test_pst(self):
        tz = _tz_offset("PST")
        dt = datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc).astimezone(tz)
        assert dt.hour == 12  # 20 UTC = 12 PST

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown timezone"):
            _tz_offset("INVALID_TZ")

    def test_utc_plus(self):
        tz = _tz_offset("UTC+5")
        dt = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc).astimezone(tz)
        assert dt.hour == 17

    def test_utc_minus(self):
        tz = _tz_offset("UTC-3")
        dt = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc).astimezone(tz)
        assert dt.hour == 9


class TestWithinBlocks:
    def test_inside_block(self):
        tz = timezone.utc
        start = datetime(2026, 1, 15, 10, 0, tzinfo=tz)
        end = datetime(2026, 1, 15, 11, 0, tzinfo=tz)
        assert _within_blocks(start, end, ["09:00-17:00"]) is True

    def test_outside_block(self):
        tz = timezone.utc
        start = datetime(2026, 1, 15, 18, 0, tzinfo=tz)
        end = datetime(2026, 1, 15, 19, 0, tzinfo=tz)
        assert _within_blocks(start, end, ["09:00-17:00"]) is False

    def test_spanning_block_boundary(self):
        tz = timezone.utc
        start = datetime(2026, 1, 15, 16, 30, tzinfo=tz)
        end = datetime(2026, 1, 15, 17, 30, tzinfo=tz)
        assert _within_blocks(start, end, ["09:00-17:00"]) is False

    def test_cross_midnight(self):
        tz = timezone.utc
        start = datetime(2026, 1, 15, 23, 0, tzinfo=tz)
        end = datetime(2026, 1, 16, 1, 0, tzinfo=tz)
        assert _within_blocks(start, end, ["22:00-02:00"]) is True

    def test_multiple_blocks(self):
        tz = timezone.utc
        start = datetime(2026, 1, 15, 14, 0, tzinfo=tz)
        end = datetime(2026, 1, 15, 15, 0, tzinfo=tz)
        assert _within_blocks(start, end, ["09:00-12:00", "13:00-17:00"]) is True


class TestParseUtc:
    def test_basic_parse(self):
        dt = _parse_utc("2026-01-15T14:00Z")
        assert dt.hour == 14
        assert dt.tzinfo == timezone.utc
