"""
Root ASGI entrypoint for Vercel Serverless deployment.
Re-exports FastAPI `app` from `backend.app`.
"""

from backend.app import app

__all__ = ["app"]
