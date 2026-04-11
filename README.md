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

**The only calendar scheduling environment where agents must manage priority cascades, partial observability, and dynamically injected follow-up requests — not just pick an open slot.**

An OpenEnv RL environment that simulates the full EA/ops scheduling lifecycle: coordinate meetings across four timezones, navigate priority-driven bump chains, handle emergency follow-ups injected mid-episode, and submit a conflict-free calendar that respects every participant's working hours and preferences.

9 real-world scheduling scenarios across 3 difficulty tiers. 8 action types. Partial observability on HARD (preferred hours hidden until explicitly investigated). 64K+ unique instances per scenario via seed-based randomization. Decomposed 7-component reward with per-step transparency.

Fully deterministic reward. Zero LLM calls at runtime.

---

## What makes this environment unique

1. **Real scheduling patterns from ops practice.** Scenarios are drawn from situations that actually happen: exec-heavy urgent syncs where the only available slot requires a 3-step bump chain, IST participants whose overlapping working-hour window is just 90 minutes wide, dynamic follow-up requests arriving mid-episode after an initial triage call.

2. **Priority-driven bump mechanics.** Higher-priority meetings can displace lower-priority ones, which automatically re-enters the bumped event into the pending queue. The agent must track the cascade and re-slot bumped events before their deadlines. Wrong bump order causes unresolvable downstream conflicts.

3. **Partial observability on HARD.** Participant preferred hours are hidden until the agent explicitly calls `InspectParticipant`. Acting without investigating incurs avoidable preference penalties. The optimal strategy is to inspect before scheduling — exactly what a competent EA does.

4. **Dynamic follow-up requests mid-episode.** On MEDIUM and HARD scenarios, additional meeting requests arrive partway through the episode (simulating a manager escalating during a scheduling session). The agent must absorb the new request and re-plan without invalidating already-scheduled events.

5. **8-action investigation / scheduling toolkit.** Beyond schedule/reschedule, the agent can probe conflicts before committing (`ListConflicts`), reveal participant details (`InspectParticipant`), query the scoring policy (`GetPolicy`), and undo the last action (`UndoLastAction`). The action space mirrors a real scheduling assistant's capability set.

6. **Decomposed 7-component reward with per-step transparency.** Every step returns `last_reward_components`. The terminal `reward_breakdown` names exactly which constraints were violated and by how much. An agent — and a developer — can audit every point of every episode.

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

> **Note:** Live model inference baselines are pending Space deployment and will be added here after runs against the hosted endpoint. The table below shows the **oracle upper-bound scores** produced by `scripts/verify_scenarios.py` — the maximum a perfect agent can achieve on each scenario given the known-optimal action sequence. These define the scoring ceiling before preference penalties.

| Scenario | Tier | Oracle Score | Unavoidable Penalty | Notes |
|---|---|---|---|---|
| `EASY` — The Empty Slate | easy | **0.989** | — | UTC participants, shared prefs. Near-perfect achievable. |
| `EASY_B` — The Timezone Overlap | easy | **0.883** | −0.10 pref | EST/PST overlap window is outside both preferred hours. |
| `EASY_C` — The Lunch Break Gap | easy | **0.917** | −0.07 pref | IST-offset constraint; preferred window narrowly satisfied. |
| `MEDIUM` — The Greedy Preference Trap | medium | **0.807** | −0.15 comp. | Only 2/3 requests fully in preferred range. Greedy path: ~0.65. |
| `MEDIUM_B` — The Blocker Sandwich | medium | **≥0.75** | — | Pre-block gap 09:00–14:00Z. Greedy agents try 14:00Z and fail. |
| `MEDIUM_C` — The Bump Chain | medium | **0.952** | — | Bumped event re-slotted cleanly. Near-perfect if bump handled. |
| `HARD` — The Zero-Sum Domino Cascade | hard | **0.799** | −0.20 pref | Inspect-first path recovers most preference penalty. |
| `HARD_B` — The VIP No-Bump Gridlock | hard | **0.690** | −0.175 comp. | CEO+Alice routing around blockers leaves partial completion. |
| `HARD_C` — The Decoy Trap | hard | **0.792** | −0.05 pref | Inspect Dev before acting avoids the decoy trap. |

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
    print(f"Score: {final.observation.score}")
    print(f"Breakdown: {final.observation.reward_breakdown}")
```

---

## Action Space (8 commands)

| Command | Parameters | Purpose |
|---------|-----------|---------|
| `CheckAvailability` | `target_id`, `proposed_start_utc` | Probe a slot without committing. Returns constraint violations and preference notes. |
| `ScheduleNew` | `target_id`, `proposed_start_utc` | Schedule a pending request. Auto-bumps lower-priority conflicts. |
| `RescheduleExisting` | `target_id`, `proposed_start_utc` | Move an already-scheduled event. Auto-bumps lower-priority conflicts. |
| `SubmitFinalCalendar` | — | Finalize and score the episode. Triggers 7-component reward computation. |
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
    reward_breakdown: Optional[Dict[str, float]]   # Terminal 7-component breakdown
    available_commands: List[str]                  # Valid commands this step
    investigation_budget_remaining: int            # Remaining InspectParticipant budget
    total_requests_seen: int                       # Includes dynamically injected requests
    requests_completed: int
```

---

## Reward Function

### 7 components (max = 1.00, clamped to [0.01, 0.99])

| # | Component | Weight | What it measures |
|---|-----------|--------|-----------------|
| 1 | Completion | 0.35 | Fraction of all requests scheduled |
| 2 | Deadline Compliance | 0.20 | Fraction of meetings finishing before their deadline |
| 3 | Working Hours | 0.15 | Fraction of meetings within every attendee's working hours |
| 4 | Preference Quality | 0.10 | Fraction of meetings within preferred hours |
| 5 | Conflict Avoidance | 0.10 | Penalizes double-bookings in final calendar |
| 6 | Efficiency | 0.05 | `optimal_steps / actual_turns`, scaled by completion |
| 7 | Investigation Discipline | 0.05 | Fraction of HARD participants inspected before scheduling (HARD only) |

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

### Medium (3 scenarios) — multi-request, preference traps, dynamic followups

| ID | Name | Catch | Max Score |
|---|---|---|---|
| `MEDIUM` | The Greedy Preference Trap | Priya has a 7-hour morning blocker. Greedy first-slot scheduling incurs preference penalties. Dynamic follow-up injected at step 6. | 0.90 |
| `MEDIUM_B` | The Blocker Sandwich | Urgent 3-party meeting must fit between two blockers (high + medium priority). `ListConflicts` is essential. | 0.88 |
| `MEDIUM_C` | The Bump Chain | Urgent meeting must displace low-priority Bob event, then bumped event must be re-slotted before 17:00 deadline. | 0.88 |

### Hard (3 scenarios) — partial observability, cascades, traps, dynamic followups

| ID | Name | Catch | Max Score |
|---|---|---|---|
| `HARD` | The Zero-Sum Domino Cascade | 5-step backward-planned cascade across 3 priority levels. Emergency debrief injected mid-episode. Preferred hours hidden. | 0.80 |
| `HARD_B` | The VIP No-Bump Gridlock | CEO has a 3-hour unbumpable VIP block. Agent must route around both CEO and Alice's blockers. Preferred hours hidden. | 0.80 |
| `HARD_C` | The Decoy Trap | Low-priority decoy event at 15:00 can be bumped, but doing so blocks the critical urgent slot for Dev. Must inspect Dev before acting. | 0.78 |

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

- **Environment server**: FastAPI with WebSocket session isolation, max concurrent sessions unbounded
- **Action engine**: 8-command handler with constraint evaluation, priority-ordered bump mechanics, undo stack
- **Scenarios**: 9 hardened scheduling puzzles (3 per tier), all in `server/scenarios.py`
- **Randomization**: Seed-based name/title/deadline substitution in `server/scenario_resolver.py`
- **Reward**: 7-component decomposed, fully deterministic, in `server/reward.py`
- **Graders**: Per-tier `grade_easy/medium/hard` in `server/graders.py`, referenced in `openenv.yaml`
- **Client**: `MeetingNegotiatorV1Env` (async WebSocket)
- **Tests**: 46 pytest tests across 5 modules — lifecycle, reward, timezone, graders, randomization

## Project Structure

```
meeting_negotiator_v1/
├── openenv.yaml              # OpenEnv spec: version, tasks, grader references
├── inference.py              # LLM agent with Phase 2 structured logging
├── client.py                 # Async WebSocket client
├── models.py                 # Pydantic models: Action, Observation, State
├── tests/
│   ├── test_env_lifecycle.py # Reset, step, new actions, observability, followups
│   ├── test_reward.py        # 7-component scoring
│   ├── test_timezone.py      # IST, cross-midnight, block boundary
│   ├── test_graders.py       # grade_easy/medium/hard determinism
│   └── test_randomization.py # Seed determinism, structural invariance
└── server/
    ├── app.py                        # FastAPI entrypoint
    ├── meeting_negotiator_v1_environment.py  # Core engine (action handlers)
    ├── scenarios.py                  # 9 scenario definitions + registry
    ├── scenario_resolver.py          # Seed-based anti-memorization
    ├── reward.py                     # 7-component decomposed reward
    └── graders.py                    # Per-task grader functions
```
