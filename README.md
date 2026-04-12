---
title: Meeting Negotiator V1 Environment
emoji: "📅"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - scheduling
  - calendar
  - real-world
  - negotiation
---

# Meeting Negotiator V1 Environment

**A calendar scheduling environment where agents must manage priority cascades, protected-event escalations, partial observability, and recovery work — not just pick an open slot.**

An OpenEnv RL environment that simulates the full EA/ops scheduling lifecycle: coordinate meetings across four timezones, navigate priority-driven bump chains, handle protected-event recovery requests and hard-tier mid-episode pressure, and submit a conflict-free calendar that respects every participant's working hours and preferences.

9 real-world scheduling scenarios across 3 difficulty tiers. 8 action types. Partial observability on HARD (preferred hours hidden until explicitly investigated). 64K+ unique instances per scenario via seed-based randomization. Decomposed terminal reward with stability penalties and recovery credit.

Fully deterministic reward. Zero LLM calls at runtime.

---

## What makes this environment unique

1. **Real scheduling patterns from ops practice.** Scenarios are drawn from situations that actually happen: exec-heavy urgent syncs where the only available slot requires a protected-event cascade, IST participants whose overlapping working-hour window is just 90 minutes wide, and reschedule decisions that create explicit recovery work.

2. **Priority-driven bump mechanics.** Higher-priority meetings can displace lower-priority ones, which automatically re-enters the bumped event into the pending queue. The agent must track the cascade and re-slot bumped events before their deadlines. Wrong bump order causes unresolvable downstream conflicts.

3. **Partial observability on HARD.** Participant preferred hours are hidden until the agent explicitly calls `InspectParticipant`. Acting without investigating incurs avoidable preference penalties. The optimal strategy is to inspect before scheduling — exactly what a competent EA does.

4. **Protected-event escalations and recovery requests.** Disturbing certain structural meetings injects explicit follow-up work (`REQ-MEDB-RECOVERY`, `REQ-MEDC-RECOVERY`, `REQ-HARDB-EXEC-RECOVERY`, `REQ-HARDC-DECISION-RECOVERY`). The agent must absorb the recovery work and re-plan without invalidating already-scheduled events.

5. **8-action investigation / scheduling toolkit.** Beyond schedule/reschedule, the agent can probe conflicts before committing (`ListConflicts`), reveal participant details (`InspectParticipant`), query the scoring policy (`GetPolicy`), and undo the last action (`UndoLastAction`). The action space mirrors a real scheduling assistant's capability set.

6. **Decomposed reward with per-step transparency.** Every step returns `last_reward_components`. The terminal `reward_breakdown` includes completion, deadline, working-hours, preference, conflict, efficiency, investigation, `stability_penalty`, and `recovery_credit`. An agent — and a developer — can audit every point of every episode.

### Example: `HARD — The Zero-Sum Domino Cascade`

The hardest canonical scenario. CEO, Alice, and Bob need an urgent all-hands before 15:00 UTC, but Bob is fully blocked 08:00–18:00 UTC with urgent events:

```
   REQ-URGENT-ALL-HANDS (CEO + Alice + Bob, deadline 15:00Z)
   REQ-CTO-SYNC (CEO + CTO + Alice, deadline 22:00Z)
   REQ-HARD-FU1 [INJECTED MID-EPISODE] (CEO + Alice, urgent)

   Calendar (Bob):  08:00─14:00 URGENT  ·  14:00─15:00 URGENT  ·  15:00─16:00 URGENT  ·  17:00─18:00 URGENT
   Calendar (Alice):              14:00─15:00 URGENT  ·  15:00─16:00 URGENT  ·  16:00─17:00 URGENT+Dev
```

**What the agent must figure out (optimal path):**
- Bob's calendar is packed — but `EVT-ALICE-DEV-URGENT` at 16:00 can be shifted to 17:00 (bumps the low-priority Dev async event).
- That frees 16:00 for `EVT-ALICE-BOB-URGENT`.
- That frees 14:00 completely → schedule `REQ-URGENT-ALL-HANDS` at 14:00Z.
- The bumped Dev event must be re-slotted before 18:00Z.
- Emergency debrief injected mid-episode must fit CEO + Alice after their schedules clear.

Greedy agents that try to schedule `REQ-URGENT-ALL-HANDS` directly hit a blocking cascade and fail. The correct solution requires 5 steps with backward planning across 3 cascade levels.

---

## Oracle-Verified Score Bounds

> **Note:** Live model inference baselines are still limited. The table below shows the current **verified reference-plan scores** from `scripts/verify_scenarios.py` for the repository state on disk. These are known-good baseline plans, not formal optimality proofs.

| Scenario | Tier | Oracle Score | Unavoidable Penalty | Notes |
|---|---|---|---|---|
| `EASY` — The Empty Slate | easy | **0.989** | — | UTC participants, shared prefs. Near-perfect achievable. |
| `EASY_B` — The Timezone Overlap | easy | **0.883** | −0.10 pref | EST/PST overlap window is outside both preferred hours. |
| `EASY_C` — The Lunch Break Gap | easy | **0.917** | −0.07 pref | IST-offset constraint; preferred window narrowly satisfied. |
| `MEDIUM` — The Greedy Preference Trap | medium | **0.948** | −0.04 pref. | Best path is `REQ-MED-2 @ 16:00Z` then `REQ-MED-1 @ 17:00Z`; greedy ordering burns the better handoff slot. |
| `MEDIUM_B` — The Blocker Sandwich | medium | **0.923** | stability −0.02 | Day-2 placement at `14:00Z` displaces a protected placeholder and requires explicit recovery work. |
| `MEDIUM_C` — The Bump Chain | medium | **0.935** | stability −0.02 | Urgent placement bumps Bob's low event and spawns a recovery task that must be cleared. |
| `HARD` — The Zero-Sum Domino Cascade | hard | **0.504** | large pref/coverage loss | Current hard path is intentionally partial-credit and demands backward planning through a dense cascade. |
| `HARD_B` — The 60-Minute VIP Bottleneck | hard | **0.878** | stability −0.02 | Urgent 3-way sync now bumps the protected CEO block, creating both a bumped replacement request and an executive recovery sync. |
| `HARD_C` — The 48-Hour Blind Cascade | hard | **0.655** | stability −0.02 | The protected trap block can be displaced only by accepting recovery work on the same day. |

**To run the oracle audit locally:**
```bash
python scripts/verify_scenarios.py              # all 9 scenarios
python scripts/verify_scenarios.py --tier hard  # HARD tier only
python scripts/verify_scenarios.py --id EASY_C  # single scenario
```

**To generate live model baselines** (requires a running Space or local server):
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-key"
python inference.py --space https://your-space.hf.space
```

---

## Quick Start

```python
from meeting_negotiator_v1 import MeetingNegotiatorV1Action, MeetingNegotiatorV1Env

async with MeetingNegotiatorV1Env(base_url="https://your-space.hf.space") as env:
    obs = await env.reset(scenario_id="HARD", seed=42)

    # 1. Investigate before acting (critical on HARD)
    obs = await env.step(MeetingNegotiatorV1Action(
        command="InspectParticipant",
        target_id="CEO",
    ))
    print(obs.last_action_feedback)  # Reveals hidden preferred hours

    # 2. Check slot viability before committing
    obs = await env.step(MeetingNegotiatorV1Action(
        command="ListConflicts",
        target_id="REQ-URGENT-ALL-HANDS",
        proposed_start_utc="2026-01-15T14:00Z",
    ))

    # 3. Schedule (may auto-bump lower-priority events)
    obs = await env.step(MeetingNegotiatorV1Action(
        command="RescheduleExisting",
        target_id="EVT-ALICE-DEV-URGENT",
        proposed_start_utc="2026-01-15T17:00Z",
    ))

    # 4. Submit when all requests are scheduled
    final = await env.step(MeetingNegotiatorV1Action(command="SubmitFinalCalendar"))
    print(f"Score: {final.score}")
    print(f"Breakdown: {final.reward_breakdown}")
```

---

## Action Space (8 commands)

| Command | Parameters | Purpose |
|---------|-----------|---------|
| `CheckAvailability` | `target_id`, `proposed_start_utc` | Probe a slot without committing. Returns constraint violations and preference notes. |
| `ScheduleNew` | `target_id`, `proposed_start_utc` | Schedule a pending request. Auto-bumps lower-priority conflicts. |
| `RescheduleExisting` | `target_id`, `proposed_start_utc` | Move an already-scheduled event. Auto-bumps lower-priority conflicts. |
| `SubmitFinalCalendar` | — | Finalize and score the episode. Triggers the terminal reward computation with stability and recovery terms. |
| `InspectParticipant` | `target_id` | Reveal participant details including hidden preferred hours (costs investigation budget). |
| `ListConflicts` | `target_id`, `proposed_start_utc` | List all conflicting events at a proposed slot with bumpability tags. |
| `GetPolicy` | — | Print the full scoring policy and per-step penalty/bonus table. |
| `UndoLastAction` | — | Reverse the last `ScheduleNew` or `RescheduleExisting`. |

**Investigation budget:** On HARD, `InspectParticipant` draws from a per-episode budget. Over-investigation is penalized; under-investigation (acting blind) incurs preference penalties.

---

## Observation Space

```python
class MeetingNegotiatorV1Observation(BaseModel):
    current_time_utc: str                          # Current episode time
    participants: Dict[str, Participant]           # Timezones, working hours, preferred hours
    calendar_state: List[ScheduledEvent]           # Already-scheduled meetings
    pending_requests: List[MeetingRequest]         # Meetings still to be scheduled
    last_action_feedback: str                      # Detailed per-attendee outcome message
    turn_count: int
    max_turns: int
    score: Optional[float]                         # None until SubmitFinalCalendar
    reward: float                                  # Cumulative step reward
    done: bool
    last_reward_components: Dict[str, float]       # Per-step reward decomposition
    reward_breakdown: Optional[Dict[str, float]]   # Terminal reward breakdown incl. stability/recovery
    available_commands: List[str]                  # Valid commands this step
    investigation_budget_remaining: int            # Remaining InspectParticipant budget
    total_requests_seen: int                       # Includes triggered recovery follow-ups
    requests_completed: int
```

---

## Reward Function

### Base components + stability/recovery adjustments (clamped to [0.01, 0.99])

| # | Component | Weight | What it measures |
|---|-----------|--------|-----------------|
| 1 | Completion | 0.35 | Fraction of all requests scheduled |
| 2 | Deadline Compliance | 0.20 | Fraction of meetings finishing before their deadline |
| 3 | Working Hours | 0.15 | Fraction of meetings within every attendee's working hours |
| 4 | Preference Quality | 0.10 | Fraction of meetings within preferred hours |
| 5 | Conflict Avoidance | 0.10 | Penalizes double-bookings in final calendar |
| 6 | Efficiency | 0.05 | `optimal_steps / actual_turns`, scaled by completion |
| 7 | Investigation Discipline | 0.05 | Fraction of key HARD participants inspected before or during the scheduling chain |
| 8 | Stability Penalty | variable | Harmful transitions after disturbing protected meetings |
| 9 | Recovery Credit | variable | Credit for resolving recovery work spawned by cascades |

**Per-step penalties** (dense reward during episode):

| Event | Penalty |
|-------|---------|
| Step cost | −0.01 |
| Invalid action | −0.05 |
| Conflict attempt (blocking priority) | −0.03 |
| Preference violation (per attendee) | −0.02 |

**Per-step bonuses:**

| Event | Bonus |
|-------|-------|
| Successful `ScheduleNew` | +0.10 |
| Successful `RescheduleExisting` | +0.05 |
| Priority bonus (urgent > medium > low) | up to +0.08 |
| Early deadline (>30 min margin) | +0.03 |

---

## Anti-Memorization via Seed-Based Randomization

Each scenario supports `reset(seed=N)`. Surface features (participant names, meeting titles, deadline jitter) are sampled from pools per reset, while the structural puzzle (attendee set topology, priority ordering, bump cascade sequence, deadline window) remains invariant.

**What randomizes:** participant names (pools of 5 per name), meeting titles (pools of 4 per title), deadline times (±15 min forward jitter).

**What stays fixed:** attendee relationships, priority ordering, bump cascade structure, working hour blocks, timezone assignments.

**Backward compatibility:** `seed=None` (default) returns the canonical scenario unchanged.

---

## Scenarios (9)

### Easy (3 scenarios) — no calendar blockers, single timezone overlap

| ID | Name | Catch | Max Score |
|---|---|---|---|
| `EASY` | The Empty Slate | Single 1:1 UTC meeting, no blockers. Tests basic constraint parsing. | 0.99 |
| `EASY_B` | The Timezone Overlap | EST/PST participants with non-overlapping preferred windows. Agent must find the UTC intersection window. | 0.95 |
| `EASY_C` | The Lunch Break Gap | 3-party meeting with IST participant. Working hours cross the Alice/Bob lunch gap. IST offset = 5.5h. | 0.95 |

### Medium (3 scenarios) — multi-request, preference traps, protected recovery work

| ID | Name | Catch | Max Score |
|---|---|---|---|
| `MEDIUM` | The Greedy Preference Trap | Two-request ordering puzzle: `REQ-MED-2 @ 16:00Z` and `REQ-MED-1 @ 17:00Z` is the good path; greedy first-slot ordering loses preference quality. | 0.95 |
| `MEDIUM_B` | The Blocker Sandwich | Urgent 3-party meeting only works on day 2 by displacing a protected hold and then resolving explicit recovery work. | 0.93 |
| `MEDIUM_C` | The Bump Chain | Urgent meeting displaces Bob's low-priority event and also creates a separate recovery obligation that must be scheduled. | 0.94 |

### Hard (3 scenarios) — partial observability, protected cascades, recovery chains

| ID | Name | Catch | Max Score |
|---|---|---|---|
| `HARD` | The Zero-Sum Domino Cascade | Backward-planned cascade across 3 priority levels with hidden preferences and narrow partial-credit margins. | 0.51 |
| `HARD_B` | The 60-Minute VIP Bottleneck | `REQ-HARDB-ALL @ 17:00Z` can displace the protected CEO block, but that creates both a bumped replacement request and an executive recovery sync. | 0.88 |
| `HARD_C` | The 48-Hour Blind Cascade | Day-1 urgent board prep only fits by displacing a protected trap block, which forces same-day recovery work under hidden preference pressure. | 0.66 |

---

## Run Locally

```bash
git clone https://huggingface.co/spaces/your-username/meeting-negotiator-v1
cd meeting_negotiator_v1
uv sync
uv run server
```

## Run Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-api-key"

# Default: all 9 scenarios
python inference.py --space https://your-username-meeting-negotiator-v1.hf.space

# Single tier
python inference.py --space https://your-username-meeting-negotiator-v1.hf.space --tier HARD

# With randomization (seed per episode)
SEED=42 python inference.py --space https://your-username-meeting-negotiator-v1.hf.space
```

Inference output follows the Hackathon Phase 2 structured format:

```
[START] task=HARD
[STEP] step=1 reward=-0.01
[STEP] step=2 reward=0.09
...
[END] task=HARD score=0.7832 steps=11
```

---

## Architecture

- **Environment server**: FastAPI with WebSocket session isolation, `max_concurrent_envs=1` in the current app config
- **Action engine**: 8-command handler with constraint evaluation, priority-ordered bump mechanics, undo stack
- **Scenarios**: 9 hardened scheduling puzzles (3 per tier), all in `server/scenarios.py`
- **Randomization**: Seed-based name/title/deadline substitution in `server/scenario_resolver.py`
- **Reward**: decomposed deterministic scorer with stability/recovery terms, in `server/reward.py`
- **Graders**: Per-tier `grade_easy/medium/hard` in `server/graders.py`, referenced in `openenv.yaml`
- **Client**: `MeetingNegotiatorV1Env` (async WebSocket)
- **Tests**: 52 pytest tests across lifecycle, reward, timezone, graders, and randomization coverage

## Project Structure

```
meeting_negotiator_v1/
├── openenv.yaml              # OpenEnv spec: version, tasks, grader references
├── inference.py              # LLM agent with Phase 2 structured logging
├── client.py                 # Async WebSocket client
├── models.py                 # Pydantic models: Action, Observation, State
├── tests/
│   ├── test_env_lifecycle.py # Reset, step, new actions, observability, followups
│   ├── test_reward.py        # terminal scoring + stability/recovery checks
│   ├── test_timezone.py      # IST, cross-midnight, block boundary
│   ├── test_graders.py       # grade_easy/medium/hard determinism
│   └── test_randomization.py # Seed determinism, structural invariance
└── server/
    ├── app.py                        # FastAPI entrypoint
    ├── meeting_negotiator_v1_environment.py  # Core engine (action handlers)
    ├── scenarios.py                  # 9 scenario definitions + registry
    ├── scenario_resolver.py          # Seed-based anti-memorization
    ├── reward.py                     # decomposed terminal reward
    └── graders.py                    # Per-task grader functions
```
