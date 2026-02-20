"""Structured logging with structlog + rich."""

from __future__ import annotations

import logging
import sys

import structlog
from rich.console import Console

from slm_council.config import settings

_console = Console(stderr=True)


def setup_logging() -> None:
    """Configure structlog for the entire application."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named logger."""
    return structlog.get_logger(name)


def truncate_for_log(value: str, max_chars: int) -> str:
    """Truncate long text fields for structured logging output."""
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}... <truncated {len(value) - max_chars} chars>"
