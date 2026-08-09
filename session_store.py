"""
Session storage abstraction.

Local dev / traditional hosting (no env vars set) -> plain in-memory dict.
Vercel / any stateless serverless host -> Redis via REST (Vercel KV or
Upstash), since each invocation may hit a fresh instance with no shared
memory.

Auto-detects which to use based on environment variables, so main.py doesn't
need to know or care which backend is active.
"""
import json
import os
from typing import Optional

_KV_URL = os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL")
_KV_TOKEN = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN")

USE_REDIS = bool(_KV_URL and _KV_TOKEN)

TTL_SECONDS = 60 * 60 * 4  # sessions expire after 4 hours of inactivity

if USE_REDIS:
    from upstash_redis import Redis
    _redis = Redis(url=_KV_URL, token=_KV_TOKEN)
else:
    _MEMORY: dict = {}


def load(session_id: str) -> Optional[dict]:
    """Return the stored session dict, or None if it doesn't exist."""
    if USE_REDIS:
        raw = _redis.get(session_id)
    else:
        raw = _MEMORY.get(session_id)
    return json.loads(raw) if raw else None


def save(session_id: str, data: dict) -> None:
    """Persist the session dict."""
    raw = json.dumps(data)
    if USE_REDIS:
        _redis.set(session_id, raw, ex=TTL_SECONDS)
    else:
        _MEMORY[session_id] = raw
