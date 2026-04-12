"""Standalone grader functions for OpenEnv task evaluation.

Three tier graders (`grade_easy`, `grade_medium`, `grade_hard`) referenced from
`openenv.yaml`. Multiple scenario tasks share the same grader; each call must
receive episode `scenario_id` (top-level arg or in **kwargs from `state()`)
so `compute_final_score` applies correct rules (e.g. HARD* investigation).

When `scenario_id` missing (unit tests), defaults: EASY / MEDIUM / HARD.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from .reward import compute_final_score
except ImportError:
    import os as _os
    import sys as _sys
    _server_dir = _os.path.dirname(_os.path.abspath(__file__))
    if _server_dir not in _sys.path:
        _sys.path.insert(0, _server_dir)
    from reward import compute_final_score


def _to_dict_list(items: Any) -> List[Dict]:
    """Convert a list of Pydantic models or dicts to plain dicts."""
    result = []
    for item in (items or []):
        if hasattr(item, "model_dump"):
            result.append(item.model_dump())
        elif isinstance(item, dict):
            result.append(item)
    return result


def _to_participant_dict(participants: Any) -> Dict[str, Dict]:
    """Convert participants dict (may contain Pydantic models) to plain dicts."""
    if not participants:
        return {}
    result = {}
    for name, p in participants.items():
        if hasattr(p, "model_dump"):
            result[name] = p.model_dump()
        elif isinstance(p, dict):
            result[name] = p
        else:
            result[name] = {}
    return result


def _effective_scenario_id(
    explicit: Optional[str],
    default: str,
    kwargs: Dict[str, Any],
) -> str:
    raw = explicit if explicit is not None else kwargs.get("scenario_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().upper()
    return default


def _kwargs_without_scenario_id(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    out.pop("scenario_id", None)
    return out


def _grade_from_state(
    all_requests: Any = None,
    calendar_state: Any = None,
    participants: Any = None,
    turn_count: int = 0,
    max_turns: int = 15,
    inspected_participants: Optional[List[str]] = None,
    scenario_id: str = "",
    **kwargs,
) -> float:
    """Internal: compute reward for an episode from state data."""
    return compute_final_score(
        all_requests=_to_dict_list(all_requests),
        calendar_state=_to_dict_list(calendar_state),
        participants=_to_participant_dict(participants),
        turn_count=turn_count,
        max_turns=max_turns,
        inspected_participants=inspected_participants,
        scenario_id=scenario_id,
    )[0]


def grade_easy(
    all_requests: Any = None,
    calendar_state: Any = None,
    participants: Any = None,
    turn_count: int = 0,
    max_turns: int = 15,
    inspected_participants: Optional[List[str]] = None,
    scenario_id: Optional[str] = None,
    **kwargs,
) -> float:
    """Grade easy-tier scenarios (EASY, EASY_B, EASY_C, …)."""
    sid = _effective_scenario_id(scenario_id, "EASY", kwargs)
    rest = _kwargs_without_scenario_id(kwargs)
    return _grade_from_state(
        all_requests=all_requests,
        calendar_state=calendar_state,
        participants=participants,
        turn_count=turn_count,
        max_turns=max_turns,
        inspected_participants=inspected_participants,
        scenario_id=sid,
        **rest,
    )


def grade_medium(
    all_requests: Any = None,
    calendar_state: Any = None,
    participants: Any = None,
    turn_count: int = 0,
    max_turns: int = 15,
    inspected_participants: Optional[List[str]] = None,
    scenario_id: Optional[str] = None,
    **kwargs,
) -> float:
    """Grade medium-tier scenarios (MEDIUM, MEDIUM_B, MEDIUM_C, …)."""
    sid = _effective_scenario_id(scenario_id, "MEDIUM", kwargs)
    rest = _kwargs_without_scenario_id(kwargs)
    return _grade_from_state(
        all_requests=all_requests,
        calendar_state=calendar_state,
        participants=participants,
        turn_count=turn_count,
        max_turns=max_turns,
        inspected_participants=inspected_participants,
        scenario_id=sid,
        **rest,
    )


def grade_hard(
    all_requests: Any = None,
    calendar_state: Any = None,
    participants: Any = None,
    turn_count: int = 0,
    max_turns: int = 15,
    inspected_participants: Optional[List[str]] = None,
    scenario_id: Optional[str] = None,
    **kwargs,
) -> float:
    """Grade hard-tier scenarios (HARD, HARD_B, HARD_C, …)."""
    sid = _effective_scenario_id(scenario_id, "HARD", kwargs)
    rest = _kwargs_without_scenario_id(kwargs)
    return _grade_from_state(
        all_requests=all_requests,
        calendar_state=calendar_state,
        participants=participants,
        turn_count=turn_count,
        max_turns=max_turns,
        inspected_participants=inspected_participants,
        scenario_id=sid,
        **rest,
    )
