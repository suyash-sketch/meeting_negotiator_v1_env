"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""
import re
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone, time
from typing import Optional, List, Dict

from openai import AsyncOpenAI
from dotenv import load_dotenv

try:
    from ..meeting_negotiator_v1 import MeetingNegotiatorV1Action, MeetingNegotiatorV1Env
except ImportError:
    import os as _os
    import sys as _sys
    _parent_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _parent_dir not in _sys.path:
        _sys.path.insert(0, _parent_dir)
    from meeting_negotiator_v1 import MeetingNegotiatorV1Action, MeetingNegotiatorV1Env

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-oss-120b:groq"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
SEED = int(os.getenv("SEED", "0")) or None

llm_client = AsyncOpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

USE_FALLBACK = os.getenv("USE_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
USE_FALLBACK = USE_FALLBACK or not (API_BASE_URL and MODEL_NAME and API_KEY)
MAX_RETRIES = 3
BENCHMARK_NAME = "meeting_negotiator_v1"
SUCCESS_SCORE_THRESHOLD = 0.90


# ============================================================================
# STRUCTURED STDOUT LOGGING (MANDATORY FOR GRADER)
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ============================================================================
# UTILITIES & FALLBACK SOLVER
# ============================================================================

def _sanitize_command(raw: str) -> str:
    raw = (raw or "").strip().lower()
    mapping = {
        "checkavailability":    "CheckAvailability",
        "check_availability":   "CheckAvailability",
        "schedulenew":          "ScheduleNew",
        "schedule_new":         "ScheduleNew",
        "rescheduleexisting":   "RescheduleExisting",
        "reschedule_existing":  "RescheduleExisting",
        "submitfinalcalendar":  "SubmitFinalCalendar",
        "submit_final_calendar":"SubmitFinalCalendar",
        "inspectparticipant":   "InspectParticipant",
        "inspect_participant":  "InspectParticipant",
        "listconflicts":        "ListConflicts",
        "list_conflicts":       "ListConflicts",
        "getpolicy":            "GetPolicy",
        "get_policy":           "GetPolicy",
        "undolastaction":       "UndoLastAction",
        "undo_last_action":     "UndoLastAction",
    }
    return mapping.get(raw, "SubmitFinalCalendar")

def _sanitize_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip()
    if value in ("", "None", "null"):
        return None
    
    return value

def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)

def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()

def _tz_offset(tz: str) -> timezone:
    tz_map = {"PST": -8, "CST": -6, "EST": -5, "GMT": 0, "UTC": 0, "IST": 5.5}
    if tz in tz_map: return timezone(timedelta(hours=tz_map[tz]))
    if tz.startswith("UTC"):
        raw = tz[3:]
        if not raw: return timezone.utc
        sign = 1 if raw[0] == "+" else -1
        raw = raw[1:] if raw[0] in "+-" else raw
        if ":" in raw:
            hours, minutes = raw.split(":", 1)
            offset = timedelta(hours=sign * int(hours), minutes=sign * int(minutes))
        else:
            offset = timedelta(hours=sign * int(raw))
        return timezone(offset)
    return timezone.utc

def _within_blocks(local_start: datetime, local_end: datetime, blocks: List[str]) -> bool:
    for block in blocks:
        try:
            start_str, end_str = block.split("-")
            start_time = _parse_time(start_str)
            end_time = _parse_time(end_str)
        except ValueError: continue
        block_start = datetime.combine(local_start.date(), start_time, tzinfo=local_start.tzinfo)
        block_end = datetime.combine(local_start.date(), end_time, tzinfo=local_start.tzinfo)
        if block_end <= block_start: block_end += timedelta(days=1)
        if local_start >= block_start and local_end <= block_end: return True
    return False

def _participant_dict(participant) -> Dict:
    if hasattr(participant, "model_dump"): return participant.model_dump()
    if hasattr(participant, "dict"): return participant.dict()
    if isinstance(participant, dict): return participant
    return {}

def _within_working_hours(participant, start_dt: datetime, end_dt: datetime) -> bool:
    data = _participant_dict(participant)
    tz = _tz_offset(data.get("timezone", "UTC"))
    local_start = start_dt.astimezone(tz)
    local_end = end_dt.astimezone(tz)
    return _within_blocks(local_start, local_end, data.get("working_hours", []))

def _find_conflicts(calendar_state: List[Dict], attendees: List[str], start_dt: datetime, end_dt: datetime) -> List[Dict]:
    conflicts = []
    for event in calendar_state:
        if not set(attendees).intersection(event.get("attendees", [])): continue
        ev_start = _parse_utc(event.get("start_time_utc"))
        ev_end = ev_start + timedelta(minutes=event.get("duration_minutes", 0))
        if max(start_dt, ev_start) < min(end_dt, ev_end): conflicts.append(event)
    return conflicts

def _candidate_slots(obs, request: Dict) -> List[str]:
    start_dt = _parse_utc(obs.current_time_utc)
    deadline = _parse_utc(request.get("deadline_utc"))
    duration = int(request.get("duration_minutes", 30))
    slots = []
    cursor = start_dt.replace(minute=(start_dt.minute // 15) * 15, second=0, microsecond=0)
    while cursor + timedelta(minutes=duration) <= deadline and len(slots) < 30:
        end_dt = cursor + timedelta(minutes=duration)
        ok = True
        for attendee in request.get("attendees", []):
            participant = obs.participants.get(attendee, {})
            if not _within_working_hours(participant, cursor, end_dt):
                ok = False
                break
        if ok: slots.append(_format_utc(cursor))
        cursor += timedelta(minutes=15)
    return slots

def _fallback_action(obs) -> MeetingNegotiatorV1Action:
    if not obs.pending_requests: return MeetingNegotiatorV1Action(command="SubmitFinalCalendar")
    request = obs.pending_requests[0]
    candidates = _candidate_slots(obs, request.model_dump() if hasattr(request, "model_dump") else request)
    if not candidates: return MeetingNegotiatorV1Action(command="SubmitFinalCalendar")
    request_id = request.request_id if hasattr(request, "request_id") else request.get("request_id")
    for slot in candidates:
        start_dt = _parse_utc(slot)
        end_dt = start_dt + timedelta(minutes=request.duration_minutes if hasattr(request, "duration_minutes") else request.get("duration_minutes", 30))
        conflicts = _find_conflicts([e.model_dump() if hasattr(e, "model_dump") else e for e in obs.calendar_state], request.attendees if hasattr(request, "attendees") else request.get("attendees", []), start_dt, end_dt)
        if not conflicts: return MeetingNegotiatorV1Action(command="ScheduleNew", target_id=request_id, proposed_start_utc=slot)
    return MeetingNegotiatorV1Action(command="ScheduleNew", target_id=request_id, proposed_start_utc=candidates[0])


# ============================================================================
# LLM AGENT LOGIC
# ============================================================================

def _obs_to_prompt(obs, action_history: List[str]) -> str:
    def _ser(obj):
        if hasattr(obj, "model_dump"): return obj.model_dump()
        if isinstance(obj, list): return [_ser(i) for i in obj]
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        return obj

    participants = json.dumps(_ser(obs.participants), indent=2)
    calendar     = json.dumps(_ser(obs.calendar_state), indent=2)
    pending      = json.dumps(_ser(obs.pending_requests), indent=2)
    history_str  = "\n".join(action_history) if action_history else "No previous actions taken."

    return f"""You are a strict meeting scheduling AI.
        Goal: Schedule all pending requests while respecting constraints.
        
        Current time (UTC): {obs.current_time_utc}
        Turn: {obs.turn_count} / {obs.max_turns}
        Last feedback: {obs.last_action_feedback}

        ACTION HISTORY (Your past moves - DO NOT REPEAT FAILED OR REDUNDANT ACTIONS):
        {history_str}

        PARTICIPANTS:
        {participants}

        CALENDAR (already booked — do not double-book these):
        {calendar}

        PENDING REQUESTS (you must schedule all of these):
        {pending}

        COMMANDS (use exactly as shown):
        - CheckAvailability  target_id=<request_id>   proposed_start_utc=<slot>  → probe slot, no commitment
        - ScheduleNew        target_id=<request_id>   proposed_start_utc=<slot>  → schedule pending request
        - RescheduleExisting target_id=<event_id>     proposed_start_utc=<slot>  → move scheduled event
        - InspectParticipant target_id=<name>                                    → reveal hidden preferences
        - ListConflicts      target_id=<request_id>   proposed_start_utc=<slot>  → show conflicts + bumpability
        - GetPolicy                                                               → print scoring rules
        - UndoLastAction                                                          → undo last schedule/reschedule
        - SubmitFinalCalendar                                                     → finalize (only when ALL pending = empty)

        RULES:
        1. BUMP MECHANIC: ScheduleNew/RescheduleExisting with priority X auto-bumps events with priority < X.
        2. EQUAL OR HIGHER priority blockers CANNOT be bumped. Use RescheduleExisting to move the blocker first.
        3. If Last Feedback says 'Slot available', DO NOT check again — schedule immediately.
        4. Call SubmitFinalCalendar ONLY when 'Pending requests' is empty.
        5. All times must be UTC: "YYYY-MM-DDTHH:MMZ".
        6. ScheduleNew and RescheduleExisting MUST have a valid target_id and proposed_start_utc.
        7. InspectParticipant BEFORE scheduling on HARD — preferred hours are hidden.

        CRITICAL: Output ONLY a valid JSON object matching this exact schema:
        {{
          "thought": "1-2 sentences: timezone math + reason for this action + time.",
          "command": "<one of the 8 commands above>",
          "target_id": "<id or null>",
          "proposed_start_utc": "<YYYY-MM-DDTHH:MMZ or null>"
        }}
"""


async def call_llm_with_retry(prompt: str) -> dict:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system",
                     "content": "You are an advanced scheduling AI. You must respond with a single, strictly formatted JSON object. You must include your reasoning in the 'thought' key before the command."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
            }
            response = await llm_client.chat.completions.create(**kwargs)
            raw = (response.choices[0].message.content or "").strip()
            
            if not raw: raise ValueError("API returned an empty string.")

            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match: raw = match.group(0)
            return json.loads(raw)

        except json.JSONDecodeError as e:
            last_error = e
            print(f"[DEBUG] attempt {attempt + 1}: JSON parse failed — {e}", flush=True)
            await asyncio.sleep(2)
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str:
                wait = 5 * (attempt + 1)
                print(f"[DEBUG] attempt {attempt + 1}: rate limited, waiting {wait}s", flush=True)
                await asyncio.sleep(wait)
            else:
                print(f"[DEBUG] attempt {attempt + 1}: API Error — {e}", flush=True)
                await asyncio.sleep(2)

    print(f"[DEBUG] all {MAX_RETRIES} attempts failed: {last_error} — using safe default", flush=True)
    return {"command": "SubmitFinalCalendar", "target_id": None, "proposed_start_utc": None}


# ============================================================================
# MAIN EPISODE LOOP
# ============================================================================

async def run_episode(env: MeetingNegotiatorV1Env, episode_label: str):
    result = await env.reset(scenario_id=episode_label, seed=SEED)
    obs = result.observation
    done = result.done

    # MANDATORY: Log start of episode
    log_start(task=episode_label, env=BENCHMARK_NAME, model=MODEL_NAME)

    action_history = []
    rewards = []
    steps_taken = 0
    score = 0.0

    while not done:
    # if USE_FALLBACK:
    #     action = _fallback_action(obs)
    # else:
        prompt = _obs_to_prompt(obs, action_history)
        parsed = await call_llm_with_retry(prompt)
            
        command = _sanitize_command(parsed.get("command"))
        target_id = _sanitize_optional(parsed.get("target_id"))
        proposed_start_utc = _sanitize_optional(parsed.get("proposed_start_utc"))
        
        if command == "SubmitFinalCalendar":
            target_id = None
            proposed_start_utc = None
        
        action = MeetingNegotiatorV1Action(
            command=command,
            target_id=target_id,
            proposed_start_utc=proposed_start_utc,
        )

        # Record action history internally
        action_history.append(f"Turn {obs.turn_count}: {action.command} | Target: {action.target_id} | Start: {action.proposed_start_utc}")
        
        # Build action string for the Step Logger
        action_str = f"{action.command}('{action.target_id}','{action.proposed_start_utc}')"

        # Take Step
        result = await env.step(action)
        obs = result.observation
        done = result.done
        reward = result.reward

        steps_taken += 1
        rewards.append(reward)
        
        # Extract errors if the environment rejected the action
        error_msg = None
        fb_lower = obs.last_action_feedback.lower()
        if "error" in fb_lower or "failed" in fb_lower or "conflict" in fb_lower:
            error_msg = obs.last_action_feedback.replace("\n", " ")

        # MANDATORY: Log step
        log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error_msg)

    # Calculate final completion metrics
    score = float(obs.score) if obs.score is not None else 0.01

    # HARD tier reference oracle scores are ~0.76–0.79 with current terminal weights.
    threshold = 0.74 if episode_label.startswith("HARD") else SUCCESS_SCORE_THRESHOLD
    success = score >= threshold

    # MANDATORY: Log end of episode
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)



ALL_SCENARIOS = [
    "EASY", "EASY_B", "EASY_C",
    "MEDIUM", "MEDIUM_B", "MEDIUM_C",
    "HARD", "HARD_B", "HARD_C",
]

TIER_SCENARIOS = {
    "easy":   ["EASY", "EASY_B", "EASY_C"],
    "medium": ["MEDIUM", "MEDIUM_B", "MEDIUM_C"],
    "hard":   ["HARD", "HARD_B", "HARD_C"],
}


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Meeting Negotiator V1 inference runner.")
    parser.add_argument(
        "--scenarios", nargs="+", default=None, metavar="ID",
        help="Explicit scenario IDs to run (e.g. EASY HARD_C). Default: all 9.",
    )
    parser.add_argument(
        "--tier", choices=["easy", "medium", "hard"], default=None,
        help="Run all scenarios for a single tier.",
    )
    parser.add_argument(
        "--space", default=None,
        help="HF Space base URL (e.g. https://user-meeting-negotiator-v1.hf.space).",
    )
    args, _ = parser.parse_known_args()

    # Resolve which scenarios to run
    if args.scenarios:
        scenarios = [s.upper() for s in args.scenarios]
    elif args.tier:
        scenarios = TIER_SCENARIOS[args.tier]
    else:
        scenarios = ALL_SCENARIOS

    # Resolve env connection
    if IMAGE_NAME:
        print(f"[DEBUG] Booting environment from docker image: {IMAGE_NAME}", flush=True)
        env = await MeetingNegotiatorV1Env.from_docker_image(IMAGE_NAME)
    else:
        base_url = args.space or os.getenv("ENV_BASE_URL", "http://localhost:8000")
        print(f"[DEBUG] Connecting to: {base_url}", flush=True)
        env = MeetingNegotiatorV1Env(base_url=base_url)

    print(f"[DEBUG] Running {len(scenarios)} scenario(s): {', '.join(scenarios)}", flush=True)

    try:
        for scenario_id in scenarios:
            await run_episode(env, scenario_id)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

