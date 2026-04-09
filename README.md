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

# 📅 Meeting Negotiator V1 Environment
**Meeting Negotiator V1** is an [OpenEnv](https://huggingface.co/openenv)-compliant multi-step RL environment for real-world calendar coordination involving time zones, priorities, preferences, deadlines, and cascading conflicts.
> *"Schedule an urgent all-hands with the CEO, Alice, and Bob before 3 PM. Alice is offline until 2 PM, Bob is booked solid, and you need to coordinate with the Dev team in India without violating executive preferences. Go."*

Unlike simple scheduling problems, agents must:

* reason across multiple timezones (EST, PST, GMT, IST)
* handle priority constraints and unbumpable meetings
* optimize soft preferences vs hard constraints
* perform multi-step rescheduling (not single-step decisions)
---

## 🎯 Tasks

The environment supports scenario selection by name in `reset(scenario_id=...)`. If no scenario is provided, it cycles deterministically: `EASY` → `MEDIUM` → `HARD`.

| Scenario ID | Name | Difficulty | Max Score | Catch |
|---|---|---|---|---|
| `EASY` | The Empty Slate | 🟢 Easy | `0.99` | Introductory task verifying UTC boundary parsing. |
| `MEDIUM` | The Greedy Preference Trap | 🟡 Medium | `0.90` | Greedily taking the first available slot incurs a `-0.15` penalty. Optimal planning yields `-0.10`. |
| `HARD` | The Zero-Sum Domino Cascade | 🔴 Hard | `0.80` | Requires reverse-chronological spatial reasoning, fractional timezone math, and dodging a chronological decoy trap. |

> *Note: All dates are set to **January 15, 2026** to avoid DST ambiguity.*
>
> *Deadline semantics: a meeting must fully finish by its `deadline_utc`, not merely start before it.*

---

## 🧠 Deep Dive: The `HARD` Task

**Description:** A 5-step priority cascade across four timezones. The model must untangle an unbumpable gridlock using manual rescheduling to clear a strict-deadline funnel, while actively tracking dynamically bumped meetings.

### The Optimal Solution Path
To achieve the maximum possible score of `0.80`, the agent must execute the following exact sequence:

![Participants and Working Hours](participants_preferences.png)
![Possible Solution](hard_task_solution.png)

* **Step 1:** Move `EVT-ALICE-DEV-URGENT` from `16:00 UTC` to `17:00 UTC`. Because this is an urgent meeting, Dev's low-priority event (`EVT-DEV-LOW`) is automatically bumped and enters the pending requests queue.
* **Step 2:** The `16:00` block is now free. Move `EVT-ALICE-BOB-URGENT` to `16:00 UTC`.
* **Step 3:** The cascade is complete. The `14:00` slot is now completely free. Schedule `REQ-URGENT-ALL-HANDS` at `14:00 UTC`. *(Note: This incurs an unavoidable -0.15 preference penalty since it is outside all three attendees' preferred hours, which is expected by design).*
* **Step 4:** Dev's bumped meeting must be fully completed by its `18:00Z` deadline. Dev is wide open in the morning. Schedule `REQ-BUMPED-DEV` to `09:00 UTC`.
* **Step 5:** Schedule `REQ-CTO-SYNC` at `21:00 UTC`. This safely avoids the chronological decoy trap. It satisfies Alice and the CEO, but misses the CTO's preference *(an unavoidable -0.05 penalty).*

**Final Score Calculation:**
`Base (1.00) - Preference Penalty (-0.20) = 0.80 Final Score`

---

## ⚙️ Environment Design

### Observation Space
The agent receives a rich state payload containing the current timeline, participant parameters, and pending tasks.

```python
class MeetingNegotiatorV1Observation(BaseModel):
    current_time_utc: str
    participants: Dict[str, Participant]  # Timezones, working/preferred hours
    calendar_state: List[ScheduledEvent]  # Currently booked meetings
    pending_requests: List[MeetingRequest] # Meetings that need to be scheduled
    last_action_feedback: str             # Success/Error string from previous step
    turn_count: int
    max_turns: int
    score: Optional[float]
    reward: float
    done: bool
```
### Action Space
The agent interacts with the calendar using four distinct commands.Pythonclass 
```python
MeetingNegotiatorV1Action(BaseModel):
    command: Literal["CheckAvailability", "ScheduleNew", "RescheduleExisting", "SubmitFinalCalendar"]
    target_id: Optional[str]           # request_id (for new) or event_id (for existing)
    proposed_start_utc: Optional[str]  # e.g., "2026-01-15T15:00Z"

```
## 🏆 Rewards and Scoring

The environment provides **dense, penalty-based rewards across the trajectory**, culminating in a **strict, constraint-based final score** upon submission.

### 📊 Final Score Penalties

| Violation / Condition                        | Penalty |
| -------------------------------------------- | ------- |
| Request left unscheduled                     | -0.40   |
| Meeting finishes after its deadline         | -0.20   |
| Scheduled outside working hours              | -0.25   |
| Double-booked / conflict                     | -0.30   |
| Priority inversion (bumping higher priority) | -0.15   |
| Outside preferred hours (per attendee)       | -0.05   |

---

### ⚡ Step Modifiers (Dense Rewards)

* Small step cost per action
* Penalties for:

  * invalid actions
  * conflicts
* Bonuses for:

  * successful scheduling
  * respecting priority tier
  * early deadline satisfaction

The inference baseline treats `0.80` as a successful `HARD` run because that is the documented optimal score for the scenario.

---

### 🧠 Learning Signal

The preference signal is exposed via:

```python id="bz0e7y"
last_action_feedback
```

This helps RL agents understand:

* why an action failed
* where penalties come from

---

## 🚀 Quick Start



```python  

from meeting_negotiator_v1 import MeetingNegotiatorV1Action, MeetingNegotiatorV1Env
import MeetingNegotiatorV1Action, MeetingNegotiatorV1Env

with MeetingNegotiatorV1Env(base_url="http://localhost:8000") as env:
    result = env.reset(scenario_id="HARD")
    obs = result.observation
    print(obs.last_action_feedback)

    # Example: Check availability, schedule, and submit
    req_id = obs.pending_requests[0].request_id
    
    # Check slot
    env.step(MeetingNegotiatorV1Action(
        command="CheckAvailability",
        target_id=req_id,
        proposed_start_utc="2026-01-15T14:00Z",
    ))

    # Lock it in
    env.step(MeetingNegotiatorV1Action(
        command="ScheduleNew",
        target_id=req_id,
        proposed_start_utc="2026-01-15T14:00Z",
    ))

    # Finalize
    final = env.step(MeetingNegotiatorV1Action(command="SubmitFinalCalendar"))
    print(f"Final Score: {final.observation.score}") 

```

## 💻 Running Locally

### Start the Environment Server

```bash id="j3m8qk"
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

### Run the LLM Inference Agent

```bash id="8c2v9p"
python inference.py
```

---

### 🔑 Required Environment Variables

* `API_BASE_URL`
* `MODEL_NAME`
* `HF_TOKEN` (or `API_KEY`)

---

## 🐳 Deployment

### Build the Docker Image

```bash id="u9s4fa"
docker build -t meeting_negotiator_v1-env:latest -f server/Dockerfile .
```

---

### Push to Hugging Face (OpenEnv)

```bash id="v1k3zn"
openenv push
```

```
📁 Project StructurePlaintextmeeting_negotiator_v1/
├── openenv.yaml              # OpenEnv spec metadata
├── inference.py              # LLM inference script with grader-compatible logging
├── client.py                 # Thin OpenEnv client wrapper
├── models.py                 # Pydantic models (Action, Observation, State)
└── server/
    ├── app.py                # FastAPI app entrypoint (HTTP/WS)
    ├── Dockerfile            # Container build instructions
    └── meeting_negotiator_v1_environment.py  # Core environment engine & scenarios
Note: The environment supports multiple concurrent WebSocket sessions. Unknown timezones raise a validation error and cause the slot to be rejected.
```
