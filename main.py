"""
Universal entrypoint for QuRoute (SIH26137 | Visionaries for Change)
Enables Vercel, Uvicorn, and Gunicorn to import `app` directly from root.
"""
from backend.app import app

__all__ = ["app"]
