"""
Vercel Python entrypoint. Vercel's Python runtime auto-detects an ASGI `app`
object exported from a file under /api and serves it as a serverless
function. This just re-exports the real app defined in main.py at the repo
root so main.py itself stays framework-agnostic and runnable anywhere
(uvicorn locally, any other ASGI host, etc).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app  # noqa: E402,F401
