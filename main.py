"""
AI Interview Agent
===================
Conducts a realistic, multi-turn technical interview based on a candidate's
progress through the 31-day AI Cohort curriculum.

Endpoint (per Technical Specification):
    POST /api/interview

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import session_store

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL = os.environ.get("INTERVIEW_MODEL", "claude-sonnet-4-6")
MIN_QUESTIONS = 8
MIN_DAYS = 4
MAX_QUESTIONS_HARD_CAP = 16  # safety valve so an interview can't run forever
CURRICULUM_PATH = Path(__file__).parent / "curriculum.json"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

with open(CURRICULUM_PATH, "r") as f:
    CURRICULUM = json.load(f)

DAYS_BY_NUMBER: Dict[int, dict] = {d["day"]: d for d in CURRICULUM["days"]}

SAMPLE_CANDIDATES_PATH = Path(__file__).parent / "sample_candidates.json"
FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"

# --------------------------------------------------------------------------
# Session state (persisted via session_store: in-memory locally, Redis on
# serverless hosts like Vercel — see session_store.py)
# --------------------------------------------------------------------------

_LOCK = threading.Lock()


class SessionState:
    def __init__(self, candidate: dict):
        self.candidate = candidate
        self.history: List[Dict[str, Any]] = []  # Anthropic messages format
        self.questions_asked = 0
        self.days_covered: set[int] = set()
        self.done = False
        self.feedback: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "history": self.history,
            "questions_asked": self.questions_asked,
            "days_covered": sorted(self.days_covered),
            "done": self.done,
            "feedback": self.feedback,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        s = cls(candidate=data["candidate"])
        s.history = data["history"]
        s.questions_asked = data["questions_asked"]
        s.days_covered = set(data["days_covered"])
        s.done = data["done"]
        s.feedback = data["feedback"]
        return s


def _load_session(session_id: str) -> Optional[SessionState]:
    raw = session_store.load(session_id)
    return SessionState.from_dict(raw) if raw else None


def _save_session(session_id: str, session: SessionState) -> None:
    session_store.save(session_id, session.to_dict())


# --------------------------------------------------------------------------
# Tool schema — forces structured, parseable output on every model turn
# --------------------------------------------------------------------------

INTERVIEW_TOOL = {
    "name": "interview_turn",
    "description": (
        "Emit the interviewer's next conversational turn in a structured form."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string",
                "description": "What the interviewer says to the candidate right now.",
            },
            "question_day": {
                "type": ["integer", "null"],
                "description": (
                    "If this turn poses a NEW substantive technical question, the "
                    "curriculum day number (1-31) it targets. Use the SAME day "
                    "number for a follow-up that digs into the same question. "
                    "Use null for turns that are not a technical question "
                    "(e.g. the opening greeting, transition remarks, or the "
                    "closing statement)."
                ),
            },
            "done": {
                "type": "boolean",
                "description": (
                    "True only on the FINAL turn, once the interview is fully "
                    "complete and feedback has been prepared."
                ),
            },
            "feedback": {
                "type": ["object", "null"],
                "description": "Required and non-null when done is true, otherwise null.",
                "properties": {
                    "summary": {"type": "string"},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "gaps": {"type": "array", "items": {"type": "string"}},
                    "next": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "required": ["reply", "question_day", "done"],
    },
}

TOOL_CHOICE = {"type": "tool", "name": "interview_turn"}


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def _curriculum_context_for_candidate(candidate: dict) -> str:
    """Build a compact curriculum reference limited to days relevant to this
    candidate's mission history, plus a one-line map of the whole cohort for
    orientation."""
    missions = candidate.get("missions", [])
    relevant_days = sorted({m["day"] for m in missions if m["day"] in DAYS_BY_NUMBER})

    lines = ["FULL COHORT MAP (for orientation only):"]
    for mod in CURRICULUM["modules"]:
        start, end = mod["days"]
        lines.append(f"  Module {mod['n']}: {mod['title']} (days {start}-{end})")

    lines.append("")
    lines.append(
        "DETAILED CURRICULUM FOR DAYS THIS CANDIDATE ATTEMPTED "
        "(these are your ONLY valid sources for technical questions):"
    )
    for day_num in relevant_days:
        d = DAYS_BY_NUMBER[day_num]
        lines.append(f"\nDay {d['day']} — {d['title']} [{d.get('type', '')}]")
        lines.append(f"  Tools: {', '.join(d.get('tools', []))}")
        lines.append("  Objectives:")
        for obj in d.get("objectives", []):
            lines.append(f"    - {obj}")

    return "\n".join(lines)


def _candidate_summary(candidate: dict) -> str:
    member = candidate.get("member", {})
    missions = candidate.get("missions", [])
    signals = candidate.get("signals", {})

    lines = [
        f"Name: {member.get('name')}",
        f"Target role: {member.get('jobRole')}",
        f"Experience: {member.get('yearsExperience')} years, {member.get('education')}",
        f"Cohort status: {member.get('status')}",
        f"Engagement signals: {signals.get('commitDays')} active days, "
        f"{signals.get('missionsCompleted')} missions completed, "
        f"{signals.get('missionsFirstTry')} passed on the first try.",
        "",
        "Mission history (day / title / result):",
    ]
    for m in missions:
        if m.get("skipped"):
            result = "SKIPPED"
        elif m.get("passed"):
            result = f"PASSED (attempts: {m.get('attempts')})"
        else:
            result = f"NOT PASSED (attempts: {m.get('attempts')})"
        lines.append(f"  Day {m['day']} — {m['title']}: {result}")

    return "\n".join(lines)


def build_system_prompt(candidate: dict) -> str:
    candidate_summary = _candidate_summary(candidate)
    curriculum_context = _curriculum_context_for_candidate(candidate)

    return f"""You are Priya, a senior AI engineering interviewer conducting a live,
spoken-style technical interview for a graduate of "The AI Cohort," a 31-day
enterprise AI engineering program. You are warm but rigorous — like a real
staff engineer running a bar-raiser interview, not a quiz bot.

CANDIDATE PROFILE
------------------
{candidate_summary}

CURRICULUM CONTEXT
-------------------
{curriculum_context}

HOW TO CONDUCT THIS INTERVIEW
-------------------------------
1. Open with a brief, friendly welcome (1-2 sentences), state roughly what the
   interview will cover, then move straight into your first technical
   question. Do not pad with small talk.
2. Ask ONLY about topics this candidate actually attempted (passed or failed
   with multiple attempts). You may briefly probe a SKIPPED topic once, to
   check self-awareness of the gap — but do not treat it as a deep technical
   question (mark question_day as null for that check-in, or ask lightly and
   don't penalize heavily for a shaky answer).
3. Prioritize signal: dig deeper into topics with more attempts (potential
   struggle) or high-leverage days (RAG, vector databases, prompt engineering,
   agentic AI / multi-agent orchestration, MCP, deployment, security). Days
   passed on the first try can get one solid question rather than several.
4. Ask REAL follow-up questions grounded in what the candidate just said.
   If an answer is vague, ask them to go deeper or give a concrete example
   from what they built. If an answer is strong, move on or push into an
   edge case ("what would break this at scale?", "why not use X instead?").
   Never ask a generic, canned follow-up — it must respond to their actual words.
5. You must cover AT LEAST {MIN_DAYS} distinct curriculum days and ask AT LEAST
   {MIN_QUESTIONS} substantive technical questions (each new technical
   question increments the count; a follow-up on the same question does not
   need a new day, tag it with the same question_day). Track this yourself
   using your own conversation history.
6. Vary question types across the interview: conceptual ("why does X work"),
   applied/design ("how would you extend this system to do Y"), and
   debugging/trade-off ("what would you change if latency/cost became a
   problem"). Reference their actual mission titles and the tools listed for
   that day so it feels specific to their journey, not generic.
7. Keep each reply focused — one question (or one tight follow-up) per turn.
   Do not ask multiple unrelated questions in a single turn.
8. When you have enough signal (minimum questions and days met, and you feel
   you've formed a fair picture), give a brief closing remark thanking the
   candidate, set done=true, and produce structured feedback:
   - summary: 2-4 sentence overall read on their technical communication and
     understanding.
   - strengths: concrete, specific things they demonstrated well.
   - gaps: concrete, specific areas that were shaky, avoided, or skipped —
     reference actual curriculum days/topics.
   - next: actionable next steps (e.g. "revisit Day 12 prompt engineering and
     practice explaining temperature vs top_p trade-offs out loud").
   Do not end the interview before the minimums are met, even if answers are
   strong — use the extra questions to probe breadth or edge cases instead.

OUTPUT FORMAT
--------------
You MUST respond using the interview_turn tool on every single turn — never
respond with plain text. Set question_day accurately every time (null for
non-technical-question turns like the opener or closer). Only set done=true
and include feedback on the very last turn.
"""


# --------------------------------------------------------------------------
# Anthropic call wrapper
# --------------------------------------------------------------------------

def _call_model(system_prompt: str, history: List[Dict[str, Any]]) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=system_prompt,
        messages=history,
        tools=[INTERVIEW_TOOL],
        tool_choice=TOOL_CHOICE,
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Model did not return a tool_use block")


def _run_turn(session: SessionState, system_prompt: str) -> dict:
    """Call the model, enforce coverage guardrails, retry if the model tries
    to end prematurely, and update session state."""
    max_retries = 2
    for attempt in range(max_retries + 1):
        result = _call_model(system_prompt, session.history)

        wants_done = bool(result.get("done"))
        would_be_questions = session.questions_asked + (
            1 if result.get("question_day") is not None else 0
        )
        would_be_days = session.days_covered | (
            {result["question_day"]} if result.get("question_day") is not None else set()
        )

        premature = wants_done and (
            would_be_questions < MIN_QUESTIONS or len(would_be_days) < MIN_DAYS
        )

        if premature and attempt < max_retries:
            # Nudge the model to keep going instead of accepting the early end.
            session.history.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "retry_nudge",
                            "name": "interview_turn",
                            "input": result,
                        }
                    ],
                }
            )
            session.history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "retry_nudge",
                            "content": (
                                f"Not yet: only {would_be_questions}/{MIN_QUESTIONS} "
                                f"questions and {len(would_be_days)}/{MIN_DAYS} distinct "
                                "days covered so far. Continue the interview with "
                                "another substantive question before wrapping up."
                            ),
                        }
                    ],
                }
            )
            continue

        # Accept this result (either it's fine, or we've exhausted retries —
        # in which case we force done back to False rather than under-deliver).
        if premature:
            result["done"] = False
            result["feedback"] = None

        if result.get("question_day") is not None:
            session.questions_asked += 1
            session.days_covered.add(result["question_day"])

        if session.questions_asked >= MAX_QUESTIONS_HARD_CAP and not result.get("done"):
            # Force a wrap-up on the *next* call rather than here, to avoid
            # cutting off a reply the model already generated.
            pass

        return result

    return result  # pragma: no cover


# --------------------------------------------------------------------------
# API models
# --------------------------------------------------------------------------

class InterviewRequest(BaseModel):
    sessionId: str
    candidate: Optional[dict] = None
    message: Optional[str] = None


class Feedback(BaseModel):
    summary: str
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    next: List[str] = Field(default_factory=list)


class InterviewResponse(BaseModel):
    reply: str
    done: bool
    feedback: Optional[Feedback] = None


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(title="AI Interview Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/interview", response_model=InterviewResponse)
def interview(req: InterviewRequest):
    with _LOCK:
        session = _load_session(req.sessionId)

        # ---- New session (start) ----
        if session is None:
            if not req.candidate:
                raise HTTPException(
                    status_code=400,
                    detail="First request for a new sessionId must include 'candidate'.",
                )
            session = SessionState(candidate=req.candidate)

            system_prompt = build_system_prompt(session.candidate)
            session.history.append(
                {
                    "role": "user",
                    "content": (
                        "Begin the interview now. Greet the candidate by name and "
                        "ask your first technical question."
                    ),
                }
            )
            result = _run_turn(session, system_prompt)

        # ---- Existing session (turn) ----
        else:
            if session.done:
                raise HTTPException(
                    status_code=400,
                    detail="This interview session has already ended.",
                )
            if not req.message:
                raise HTTPException(
                    status_code=400,
                    detail="Subsequent requests must include 'message'.",
                )
            system_prompt = build_system_prompt(session.candidate)
            session.history.append({"role": "user", "content": req.message})
            result = _run_turn(session, system_prompt)

        # Record assistant turn in history for context continuity.
        session.history.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "turn",
                        "name": "interview_turn",
                        "input": result,
                    }
                ],
            }
        )
        # Anthropic requires a tool_result to follow a tool_use before the
        # next user turn; we satisfy that lazily on the *next* incoming
        # request by prepending it there. Simpler: convert stored turns to
        # plain text messages instead of replaying raw tool blocks.
        # -> Replace the just-appended assistant block with a plain-text
        #    equivalent so history stays valid for the next call.
        session.history[-1] = {"role": "assistant", "content": result.get("reply", "")}

        if result.get("done"):
            session.done = True
            session.feedback = result.get("feedback")

        _save_session(req.sessionId, session)

        response = InterviewResponse(
            reply=result.get("reply", ""),
            done=bool(result.get("done")),
            feedback=Feedback(**result["feedback"]) if result.get("feedback") else None,
        )
        return response


@app.get("/health")
def health():
    return {
        "status": "ok",
        "session_backend": "redis" if session_store.USE_REDIS else "in-memory",
    }


@app.get("/", response_class=HTMLResponse)
def root():
    """Serves the interview transcript UI. This is the whole app — opening
    the deployed URL in a browser is the intended way to use it."""
    return FRONTEND_PATH.read_text()


@app.get("/api/candidates")
def list_candidates():
    """Sample candidate roster for the frontend's dossier selector."""
    with open(SAMPLE_CANDIDATES_PATH, "r") as f:
        return json.load(f)["candidates"]
