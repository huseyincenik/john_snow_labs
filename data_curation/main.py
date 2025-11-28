"""Entry point shim so `uv run python main.py` works from the repo root."""

from __future__ import annotations

from src.main import app

if __name__ == "__main__":
    import uvicorn
    from config.settings import settings

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
