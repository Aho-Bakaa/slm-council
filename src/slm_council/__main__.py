"""CLI entry-point for the SLM Coding Council server."""

from __future__ import annotations

import uvicorn

from slm_council.config import settings


def main() -> None:
    uvicorn.run(
        "slm_council.server:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
