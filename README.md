---
title: Meeting Negotiator V1 Environment Server
emoji: "⌚"
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Meeting Negotiator V1 Environment

A multi-step RL environment for calendar coordination with time zones, priorities, preferences, deadlines, and cascading conflicts. The agent must propose slots, adjust, negotiate, and finalize a valid calendar while maximizing satisfaction and minimizing conflicts.

## Scenarios

The environment cycles deterministically across three scenarios on each reset. **Note:** Not all scenarios allow for a perfect `1.00` score. Some scenarios contain unavoidable soft-constraint (preference) conflicts to test tradeoff reasoning.  

### 1. EASY: The Empty Slate
* **Max Score: 1.00**
* **Description:** A simple 1:1 meeting with basic working hour constraints. An introductory task to verify the agent can correctly parse UTC boundaries.

### 2. MEDIUM: The Greedy Preference Trap
* **Max Score: 0.90**
* **Description:** Multi-timezone coordination optimizing for soft preferences. 
* **The Catch:** The agent must schedule two meetings into two valid slots. Greedily taking the first available slot incurs a `-0.15` penalty. Planning ahead and reversing the order yields the optimal `-0.10` penalty (Score: 0.90). A perfect 1.00 is mathematically impossible.

### 3. HARD: The Deadline Trap & The Decoy
* **Max Score: 1.00**
* **Description:** A complex priority cascade requiring multi-step lookahead, dynamic meeting bumping, strict deadline management, and soft-constraint optimization across three timezones.
* **The Catch:** The model must execute a flawless priority bump to schedule a tight-deadline meeting, while actively tracking dynamically injected pending requests. Furthermore, the timeline contains a chronological "Decoy Trap." Greedy models that grab the first available slot will silently violate executive preferences (Max Score: `0.90`). Elite models that simulate the whole board and look ahead will find the optimal slot and achieve a perfect `1.00`.

## Quick Start

```python
from meeting_negotiator_v1 import MeetingNegotiatorV1Action, MeetingNegotiatorV1Env

with MeetingNegotiatorV1Env(base_url="http://localhost:8000") as env:
    result = env.reset(scenario_id="HARD")
    obs = result.observation
    print(obs.last_action_feedback)

    # Example: Check availability, then schedule, then submit
    req_id = obs.pending_requests[0].request_id
    env.step(MeetingNegotiatorV1Action(
        command="CheckAvailability",
        target_id=req_id,
        proposed_start_utc="2026-04-01T15:00Z",
    ))

    env.step(MeetingNegotiatorV1Action(
        command="ScheduleNew",
        target_id=req_id,
        proposed_start_utc="2026-04-01T15:00Z",
    ))

    final = env.step(MeetingNegotiatorV1Action(command="SubmitFinalCalendar"))
    print("Score:", final.observation.score)
```

## Action Space

`MeetingNegotiatorV1Action`:
- `command`: `CheckAvailability`, `ScheduleNew`, `RescheduleExisting`, `SubmitFinalCalendar`
- `target_id`: request_id for ScheduleNew, event_id for RescheduleExisting
- `proposed_start_utc`: required for CheckAvailability, ScheduleNew, RescheduleExisting

## Observation Space

`MeetingNegotiatorV1Observation` includes:
- `current_time_utc`, `participants`, `calendar_state`, `pending_requests`
- `last_action_feedback`, `turn_count`, `max_turns`
- `score` (0.0-1.0), `reward`, `done`

## Reward and Scoring

- Reward is penalty-based per action with a small step cost plus penalties for invalid/conflicting actions.
- Final score is constraint-based in `[0.0, 1.0]` and computed on `SubmitFinalCalendar`.
- Constraints: deadlines, working hours, conflicts, priority inversions, and preference satisfaction.

## Building the Docker Image

```bash
docker build -t meeting_negotiator_v1-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

```bash
openenv push
```

This uploads the environment with the web UI at `/web`, API docs at `/docs`, and WebSocket at `/ws`.
