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

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone, time
from typing import Optional, List, Dict, Tuple

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
        sign = 1
        if raw[0] == "+":
            sign = 1
            raw = raw[1:]
        elif raw[0] == "-":
            sign = -1
            raw = raw[1:]
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
    if hasattr(participant, "model_dump"):
        return participant.model_dump()
    if hasattr(participant, "dict"):
        return participant.dict()
    if isinstance(participant, dict):
        return participant
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


def _candidate_slots(obs: "MeetingNegotiatorV1Observation", request: Dict) -> List[str]:
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


def _fallback_action(obs: "MeetingNegotiatorV1Observation") -> MeetingNegotiatorV1Action:
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
            return MeetingNegotiatorV1Action(
                command="ScheduleNew",
                target_id=request_id,
                proposed_start_utc=slot,
            )

    # If all slots conflict, try scheduling anyway at first candidate to trigger bumping.
    return MeetingNegotiatorV1Action(
        command="ScheduleNew",
        target_id=request_id,
        proposed_start_utc=candidates[0],
    )


async def run_episode(env: MeetingNegotiatorV1Env, episode_label: str):
    result = await env.reset()
    obs = result.observation
    done = result.done

    print(f"\n--- Starting Episode: {episode_label} ---")
    print(obs.last_action_feedback)

    while not done:
        if USE_FALLBACK:
            action = _fallback_action(obs)
        else:
            prompt = f"""
You are a meeting scheduling agent.

Goal: schedule all pending requests while respecting time zones, working hours, priorities, and deadlines.
You must operate in multiple steps (check, propose, adjust, reschedule, finalize) and then submit.

Current time (UTC): {obs.current_time_utc}
Participants: {obs.participants}
Calendar: {obs.calendar_state}
Pending requests: {obs.pending_requests}
Last feedback: {obs.last_action_feedback}
Turn: {obs.turn_count} / {obs.max_turns}

Respond ONLY with valid JSON in this format:
{{"command": "CheckAvailability|ScheduleNew|RescheduleExisting|SubmitFinalCalendar", "target_id": "...", "proposed_start_utc": "YYYY-MM-DDTHH:MMZ"}}
- For SubmitFinalCalendar, omit target_id and proposed_start_utc or set them to null.
- For ScheduleNew, target_id must be a pending request_id.
- For RescheduleExisting, target_id must be an event_id from calendar_state.
""".strip()

            try:
                response = await llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    extra_body={"reasoning" : {"enabled" : True}}
                )

                raw_content = response.choices[0].message.content.strip()
                if raw_content.startswith("```json"):
                    raw_content = raw_content.replace("```json", "").replace("```", "").strip()
                elif raw_content.startswith("```"):
                    raw_content = raw_content.replace("```", "").strip()

                parsed = json.loads(raw_content)
            except json.JSONDecodeError:
                parsed = {"command": "SubmitFinalCalendar"}

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

        result = await env.step(action)
        obs = result.observation
        done = result.done

    print(f"Final score: {obs.score}")


async def main():
    env = MeetingNegotiatorV1Env(base_url="http://localhost:8000")
    try:
        # The environment cycles deterministically through EASY, MEDIUM, HARD on each reset.
        await run_episode(env, "EASY")
        await run_episode(env, "MEDIUM")
        await run_episode(env, "HARD")
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
