"""Scenario parameter randomization for anti-memorization.

Implements domain randomization (Tier 3 per Procgen/LLF-Bench taxonomy):
surface features (participant names, meeting titles, exact deadlines)
are randomized per reset while the structural pattern (puzzle topology,
optimal path, bump cascades) remains invariant.

This prevents agents from memorizing instance-specific values ("always
schedule Alice at 14:00Z") and forces them to learn the scheduling
reasoning pattern instead.

Architecture: name-pool substitution
- Each scenario type defines pools of alternative names and titles
- On reset(seed != None), the resolver picks from each pool using seeded RNG
- seed=None returns the original scenario unchanged (backward compat)
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ── Name Pools ─────────────────────────────────────────────────────────

NAME_POOLS = {
    "Alice": ["Alice", "Akira", "Amara", "Anika", "Aria"],
    "Bob": ["Bob", "Bryce", "Boris", "Bao", "Blake"],
    "CEO": ["CEO", "VP-Exec", "Chief", "Director-A", "President"],
    "CTO": ["CTO", "VP-Eng", "TechLead", "Director-B", "ArchLead"],
    "Dev": ["Dev", "DevOps", "Engineer", "SRE-Lead", "Backend"],
    "Priya": ["Priya", "Priti", "Padma", "Pooja", "Pallavi"],
    "Jordan": ["Jordan", "Jamie", "Jesse", "Jules", "Jay"],
    "Alex": ["Alex", "Ash", "Avery", "Adrian", "Aubrey"],
}

TITLE_POOLS = {
    "1:1 Alice/Bob": ["1:1 Sync", "Bilateral Check-in", "Pair Review", "Quick Alignment"],
    "Full Team Sync": ["Team Standup", "Squad Huddle", "All-Hands Sync", "Sprint Review"],
    "Priya-Jordan Handoff": ["Cross-Team Handoff", "Shift Handoff", "EOD Transfer", "Context Pass"],
    "Urgent All Hands": ["Emergency Standup", "Critical Alignment", "Incident Sync", "Priority Huddle"],
    "CTO Sync": ["Leadership Sync", "Exec Alignment", "Strategy Check", "Tech Review"],
    "Dev Async Work": ["Async Catchup", "Background Task", "Offline Block", "Focus Time"],
}


def resolve_scenario(
    scenario_data: Dict[str, Any],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply parameter randomization to a scenario dict.

    Args:
        scenario_data: A dict representation of ScenarioSpec fields.
        seed: Randomization seed.
              - None: return the original scenario unchanged (backward compat).
              - Any int: deterministic randomized variant.

    Returns:
        A deep copy of the scenario dict with surface features randomized.
        The structural puzzle (event topology, priorities, durations) is unchanged.
    """
    if seed is None:
        return copy.deepcopy(scenario_data)

    rng = random.Random(seed)
    result = copy.deepcopy(scenario_data)

    # Build name substitution map
    name_subs: Dict[str, str] = {}
    used_names: set = set()
    for original, pool in NAME_POOLS.items():
        if original in result.get("participants", {}):
            available = [n for n in pool if n not in used_names]
            if available:
                chosen = rng.choice(available)
                name_subs[original] = chosen
                used_names.add(chosen)

    # Randomize titles
    title_subs: Dict[str, str] = {}
    for original, pool in TITLE_POOLS.items():
        title_subs[original] = rng.choice(pool)

    # Apply name substitutions
    if name_subs:
        result = _substitute_names(result, name_subs)

    # Apply title substitutions
    if title_subs:
        result = _substitute_titles(result, title_subs)

    # Randomize deadline within ±15 min (keeping structural validity)
    result = _jitter_deadlines(result, rng, max_jitter_minutes=15)

    return result


def _substitute_names(data: Dict[str, Any], subs: Dict[str, str]) -> Dict[str, Any]:
    """Recursively substitute participant names in scenario data."""
    # Substitute in participants dict (keys)
    if "participants" in data:
        new_participants = {}
        for name, pdata in data["participants"].items():
            new_name = subs.get(name, name)
            if isinstance(pdata, dict):
                pdata = dict(pdata)
                pdata["name"] = new_name
            elif hasattr(pdata, "model_dump"):
                pdata = pdata.model_dump()
                pdata["name"] = new_name
            new_participants[new_name] = pdata
        data["participants"] = new_participants

    # Substitute in event attendee lists
    for key in ["calendar_state", "pending_requests", "all_requests"]:
        items = data.get(key, [])
        for item in items:
            if isinstance(item, dict):
                if "attendees" in item:
                    item["attendees"] = [subs.get(a, a) for a in item["attendees"]]
            elif hasattr(item, "attendees"):
                item.attendees = [subs.get(a, a) for a in item.attendees]

    return data


def _substitute_titles(data: Dict[str, Any], subs: Dict[str, str]) -> Dict[str, Any]:
    """Substitute meeting titles."""
    for key in ["pending_requests", "all_requests"]:
        items = data.get(key, [])
        for item in items:
            if isinstance(item, dict):
                title = item.get("title", "")
                if title in subs:
                    item["title"] = subs[title]
            elif hasattr(item, "title"):
                if item.title in subs:
                    item.title = subs[item.title]
    return data


def _jitter_deadlines(
    data: Dict[str, Any],
    rng: random.Random,
    max_jitter_minutes: int = 15,
) -> Dict[str, Any]:
    """Add small jitter to deadlines without breaking structural validity.

    Only jitters forward (loosens deadline) to avoid making puzzles unsolvable.
    """
    from datetime import datetime, timedelta, timezone

    for key in ["pending_requests", "all_requests"]:
        items = data.get(key, [])
        for item in items:
            if isinstance(item, dict):
                dl = item.get("deadline_utc")
            elif hasattr(item, "deadline_utc"):
                dl = item.deadline_utc
            else:
                continue

            if dl:
                jitter = rng.randint(0, max_jitter_minutes)
                try:
                    dt = datetime.strptime(dl, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                    dt += timedelta(minutes=jitter)
                    new_dl = dt.strftime("%Y-%m-%dT%H:%MZ")
                    if isinstance(item, dict):
                        item["deadline_utc"] = new_dl
                    else:
                        item.deadline_utc = new_dl
                except ValueError:
                    pass  # skip malformed deadlines

    return data
