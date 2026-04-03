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

from meeting_negotiator_v1 import MeetingNegotiatorV1Action, MeetingNegotiatorV1Env

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

llm_client = AsyncOpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

USE_FALLBACK = os.getenv("USE_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
USE_FALLBACK = USE_FALLBACK or not (API_BASE_URL and MODEL_NAME and API_KEY)
MAX_RETRIES = 3


# ============================================================================
# UTILITIES & FALLBACK SOLVER
# ============================================================================

def _sanitize_command(raw: str) -> str:
    raw = (raw or "").strip().lower()
    if raw == "checkavailability" or raw == "check_availability":
        return "CheckAvailability"
    if raw == "schedulenew" or raw == "schedule_new":
        return "ScheduleNew"
    if raw == "rescheduleexisting" or raw == "reschedule_existing":
        return "RescheduleExisting"
    if raw == "submitfinalcalendar" or raw == "submit_final_calendar":
        return "SubmitFinalCalendar"
    return "SubmitFinalCalendar"


def _sanitize_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def _tz_offset(tz: str) -> timezone:
    tz_map = {"PST": -8, "EST": -5, "GMT": 0, "UTC": 0}
    if tz in tz_map:
        return timezone(timedelta(hours=tz_map[tz]))
    if tz.startswith("UTC"):
        raw = tz[3:]
        if not raw:
            return timezone.utc
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
        except ValueError:
            continue
        block_start = datetime.combine(local_start.date(), start_time, tzinfo=local_start.tzinfo)
        block_end = datetime.combine(local_start.date(), end_time, tzinfo=local_start.tzinfo)
        if block_end <= block_start:
            block_end += timedelta(days=1)
        if local_start >= block_start and local_end <= block_end:
            return True
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
        if not set(attendees).intersection(event.get("attendees", [])):
            continue
        ev_start = _parse_utc(event.get("start_time_utc"))
        ev_end = ev_start + timedelta(minutes=event.get("duration_minutes", 0))
        if max(start_dt, ev_start) < min(end_dt, ev_end):
            conflicts.append(event)
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
        if ok:
            slots.append(_format_utc(cursor))
        cursor += timedelta(minutes=15)
    return slots


def _fallback_action(obs) -> MeetingNegotiatorV1Action:
    if not obs.pending_requests:
        return MeetingNegotiatorV1Action(command="SubmitFinalCalendar")

    request = obs.pending_requests[0]
    candidates = _candidate_slots(obs, request.model_dump() if hasattr(request, "model_dump") else request)
    if not candidates:
        return MeetingNegotiatorV1Action(command="SubmitFinalCalendar")

    request_id = request.request_id if hasattr(request, "request_id") else request.get("request_id")
    for slot in candidates:
        start_dt = _parse_utc(slot)
        end_dt = start_dt + timedelta(minutes=request.duration_minutes if hasattr(request, "duration_minutes") else request.get("duration_minutes", 30))
        conflicts = _find_conflicts(
            [e.model_dump() if hasattr(e, "model_dump") else e for e in obs.calendar_state],
            request.attendees if hasattr(request, "attendees") else request.get("attendees", []),
            start_dt,
            end_dt,
        )
        if not conflicts:
            return MeetingNegotiatorV1Action(command="ScheduleNew", target_id=request_id, proposed_start_utc=slot)

    return MeetingNegotiatorV1Action(command="ScheduleNew", target_id=request_id, proposed_start_utc=candidates[0])


# ============================================================================
# LLM AGENT LOGIC
# ============================================================================

def _obs_to_prompt(obs, action_history: List[str]) -> str:
    """Serialize observation to clean JSON strings and inject history."""
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

ACTION HISTORY (Your past moves):
{history_str}

PARTICIPANTS:
{participants}

CALENDAR (already booked — do not double-book these):
{calendar}

PENDING REQUESTS (you must schedule all of these):
{pending}

RULES:
1. THE BUMP MECHANIC (CRITICAL): If a slot is blocked by a lower-priority meeting, DO NOT manually use RescheduleExisting on the blocker. Instead, directly use ScheduleNew for your higher-priority meeting in that slot. The system will automatically bump the blocker into your Pending Requests for you to handle later.
2. If Last Feedback says "Slot available", DO NOT check it again. Schedule it immediately.
3. Call SubmitFinalCalendar only when 'Pending requests' is empty.
4. All times must be UTC in format "YYYY-MM-DDTHH:MMZ".
5. ScheduleNew and RescheduleExisting MUST have a valid target_id.

You MUST respond with ONLY a valid JSON object. No markdown, no explanations, no code blocks.
Format required:
{{"command": "CheckAvailability|ScheduleNew|RescheduleExisting|SubmitFinalCalendar", "target_id": "<id or null>", "proposed_start_utc": "<YYYY-MM-DDTHH:MMZ or null>"}}"""


async def call_llm_with_retry(prompt: str) -> dict:
    """
    Call the LLM with up to MAX_RETRIES attempts.
    Optimized for highly chatty open-source / Hugging Face endpoints.
    """
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        # AGGRESSIVE JSON PROMPT:
                        "content": "You are a JSON-only API. You do not speak English. You speak ONLY JSON. Never provide explanations, preambles, or thoughts. Begin your response directly with '{' and end with '}'."
                    },
                    {"role": "user", "content": prompt}
                ],
                # Bumped to 1024 so if it DOES ramble, the JSON doesn't get cut off
                # "max_tokens": 1024,  
                "temperature": 0.0,  
            }

            response = await llm_client.chat.completions.create(**kwargs)
            raw = (response.choices[0].message.content or "").strip()

            if not raw:
                raise ValueError("API returned an empty string.")

            # Bulletproof Regex Extraction (Even if it rambles first)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw = match.group(0)

            return json.loads(raw)

        except json.JSONDecodeError as e:
            last_error = e
            preview = raw[:80].replace("\n", " ") if 'raw' in locals() else "None"
            print(f"  [WARN] attempt {attempt + 1}: JSON parse failed — {e} | Raw: {preview}...")
            await asyncio.sleep(2)

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str:
                wait = 5 * (attempt + 1)
                print(f"  [WARN] attempt {attempt + 1}: rate limited, waiting {wait}s")
                await asyncio.sleep(wait)
            else:
                print(f"  [WARN] attempt {attempt + 1}: API Error — {e}")
                await asyncio.sleep(2)

    print(f"  [ERROR] all {MAX_RETRIES} attempts failed: {last_error} — using safe default")
    return {"command": "SubmitFinalCalendar", "target_id": None, "proposed_start_utc": None}

# ============================================================================
# MAIN EPISODE LOOP
# ============================================================================

async def run_episode(env: MeetingNegotiatorV1Env, episode_label: str):
    result = await env.reset(scenario_id=episode_label)
    obs = result.observation
    done = result.done

    print(f"\n--- Starting Episode: {episode_label} ---")
    print(obs.last_action_feedback)

    action_history = [] 

    while not done:
        if USE_FALLBACK:
            action = _fallback_action(obs)
        else:
            # 1. Generate Prompt
            prompt = _obs_to_prompt(obs, action_history)
            
            # 2. Call LLM
            parsed = await call_llm_with_retry(prompt)

            # 3. Parse output safely
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

        # 4. Save to Memory and Execute
        action_history.append(f"Turn {obs.turn_count}: {action.command} | Target: {action.target_id} | Start: {action.proposed_start_utc}")

        print(f" Turn {obs.turn_count}: {action.command} | target={action.target_id} | start={action.proposed_start_utc}")
        result = await env.step(action)
        obs = result.observation
        done = result.done

    print(f"Final score: {obs.score}")


async def main():
    env = MeetingNegotiatorV1Env(base_url="https://suyashk13-meeting-negotiator-v1.hf.space")
    try:
        await run_episode(env, "EASY")
        await run_episode(env, "MEDIUM")
        await run_episode(env, "HARD")
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())