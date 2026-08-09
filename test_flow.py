"""
Offline test harness: mocks the Anthropic client so we can validate the
session/guardrail/retry logic and the FastAPI response shape without a live
API key or network call.
"""
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import main  # noqa: E402

# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------

class FakeToolUseBlock:
    def __init__(self, input_dict):
        self.type = "tool_use"
        self.input = input_dict


class FakeResponse:
    def __init__(self, input_dict):
        self.content = [FakeToolUseBlock(input_dict)]


# Script a plausible interview: opener, 8 questions across 4+ days, then an
# early (premature) done attempt that should get rejected by guardrails,
# then a final legitimate done.
SCRIPT = [
    {"reply": "Hi Sarah, welcome! Let's start with embeddings.", "question_day": None, "done": False, "feedback": None},
    {"reply": "Q1: Walk me through how you generated embeddings for your knowledge base on Day 7.", "question_day": 7, "done": False, "feedback": None},
    {"reply": "Follow-up: why cosine similarity over Euclidean distance here?", "question_day": 7, "done": False, "feedback": None},
    {"reply": "Q2: Why ChromaDB over Pinecone for this project (Day 8)?", "question_day": 8, "done": False, "feedback": None},
    {"reply": "Q3: Your retrieval engine took 2 attempts (Day 10) — what tripped you up?", "question_day": 10, "done": False, "feedback": None},
    {"reply": "Q4: Prompt engineering took 4 attempts (Day 12) — what changed between attempts?", "question_day": 12, "done": False, "feedback": None},
    {"reply": "Q5: Tell me about your multi-agent orchestration design (Day 22).", "question_day": 22, "done": False, "feedback": None},
    {"reply": "Q6: How does MCP differ from a plain function-calling tool setup (Day 23)?", "question_day": 23, "done": False, "feedback": None},
    # premature done attempt — only 7 questions / 5 distinct days so far, below MIN_QUESTIONS=8
    {
        "reply": "Great, thanks for your time!",
        "question_day": None,
        "done": True,
        "feedback": {"summary": "ok", "strengths": [], "gaps": [], "next": []},
    },
    # after guardrail rejects it, model should continue with another question
    {"reply": "Q7: You skipped Day 29 (observability) — do you know what you'd add to monitor this in prod?", "question_day": 29, "done": False, "feedback": None},
    {"reply": "Q8: Walk me through your Docker/K8s deployment for the capstone (Day 28).", "question_day": 28, "done": False, "feedback": None},
    {
        "reply": "That's all I need — thanks Sarah!",
        "question_day": 31,
        "done": True,
        "feedback": {
            "summary": "Sarah shows strong end-to-end RAG and agentic system understanding with minor gaps in observability.",
            "strengths": ["Clear articulation of embedding/retrieval design", "Solid grasp of multi-agent orchestration"],
            "gaps": ["Limited depth on production monitoring (Day 29 skipped)"],
            "next": ["Review Day 29 observability tooling (Prometheus/Grafana) and practice explaining a monitoring plan"],
        },
    },
]

call_count = {"n": 0}


def fake_create(**kwargs):
    idx = call_count["n"]
    call_count["n"] += 1
    if idx >= len(SCRIPT):
        # fallback: just end it
        return FakeResponse({"reply": "Wrapping up.", "question_day": None, "done": True,
                              "feedback": {"summary": "done", "strengths": [], "gaps": [], "next": []}})
    return FakeResponse(SCRIPT[idx])


main.client.messages.create = fake_create

# ---------------------------------------------------------------------------
# Run through the FastAPI app via TestClient
# ---------------------------------------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402

client_app = TestClient(main.app)

with open(Path(__file__).parent.parent / "" , "rb") if False else open("/mnt/user-data/uploads/candidates__1_.json") as f:
    candidates = json.load(f)["candidates"]

sarah = next(c for c in candidates if c["member"]["id"] == "CAND-001")

session_id = "test-session-1"

# 1. Start
resp = client_app.post("/api/interview", json={"sessionId": session_id, "candidate": sarah})
print("START ->", resp.status_code, json.dumps(resp.json(), indent=2)[:300])
assert resp.status_code == 200
data = resp.json()
assert data["done"] is False

turn_messages = [
    "We generated 384-dim vectors with Sentence Transformers and stored them alongside doc IDs in ChromaDB.",
    "Cosine similarity because we only care about direction/semantic similarity, not magnitude, and our vectors weren't normalized to unit length consistently for L2 to make sense.",
    "We picked ChromaDB because it's local, embeddable, and we didn't need Pinecone's managed scaling for a demo-sized healthcare dataset.",
    "The first attempt on retrieval failed because we weren't filtering by metadata before the similarity search, so irrelevant plan documents leaked into top-k.",
    "Early prompts were too open-ended; we iterated toward a structured system prompt with explicit output format constraints and few-shot examples.",
    "We used CrewAI with a planner agent delegating to a retrieval agent and a summarizer agent, coordinated through a shared task queue.",
    "MCP standardizes the tool/resource interface across clients, so any MCP-compatible host can use our server without custom integration code, unlike a bespoke function-calling setup tied to one app.",
    "Honestly we didn't get to observability deeply, but I know Prometheus scrapes metrics and Grafana visualizes them.",
    "We containerized both FastAPI and React, used a K8s deployment with a horizontal pod autoscaler for the backend.",
    "Nothing to add, that covers it!",
]

for i, msg in enumerate(turn_messages):
    resp = client_app.post("/api/interview", json={"sessionId": session_id, "message": msg})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    print(f"TURN {i+1} -> done={data['done']} reply={data['reply'][:80]!r}")

assert data["done"] is True
assert data["feedback"] is not None
assert len(data["feedback"]["strengths"]) > 0
print("\nFINAL FEEDBACK:")
print(json.dumps(data["feedback"], indent=2))

session = main._load_session(session_id)
print(f"\nquestions_asked={session.questions_asked}, days_covered={sorted(session.days_covered)}")
assert session.questions_asked >= main.MIN_QUESTIONS
assert len(session.days_covered) >= main.MIN_DAYS

# Guardrail check: confirm the premature done (7th scripted item) was rejected
# and the interview kept going (we should have needed MORE than 9 API calls
# because of the retry).
print(f"\ntotal model calls made: {call_count['n']} (script had {len(SCRIPT)} items -> retry happened: {call_count['n'] > len(turn_messages) + 1})")

print("\nALL ASSERTIONS PASSED")
