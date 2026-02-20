"""Abstract base classes for the Feedback Store system.

A FeedbackStore captures user interactions and quality signals
to improve agent performance over time.

Future implementations:
- SQLite-backed local store
- User edit diff tracking (what did the user change after generation?)
- Thumbs up/down per agent output
- Fine-tuning data export
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FeedbackType(str, Enum):
    """Types of feedback signals."""

    ACCEPT = "accept"  # User accepted the output as-is
    EDIT = "edit"  # User modified the output
    REJECT = "reject"  # User discarded the output
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


@dataclass
class FeedbackEntry:
    """A single feedback signal."""

    session_id: str
    agent_role: str
    feedback_type: FeedbackType
    original_output: str = ""
    edited_output: str = ""  # Only for EDIT type
    notes: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackStore(ABC):
    """Abstract base for feedback persistence."""

    @abstractmethod
    async def initialize(self) -> None:
        """Set up the feedback store."""
        ...

    @abstractmethod
    async def record(self, entry: FeedbackEntry) -> None:
        """Store a feedback entry."""
        ...

    @abstractmethod
    async def query(
        self,
        agent_role: str | None = None,
        feedback_type: FeedbackType | None = None,
        limit: int = 50,
    ) -> list[FeedbackEntry]:
        """Retrieve feedback entries with optional filters."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
        ...


class NullFeedbackStore(FeedbackStore):
    """No-op implementation when no feedback store is configured."""

    async def initialize(self) -> None:
        pass

    async def record(self, entry: FeedbackEntry) -> None:
        pass

    async def query(
        self,
        agent_role: str | None = None,
        feedback_type: FeedbackType | None = None,
        limit: int = 50,
    ) -> list[FeedbackEntry]:
        return []

    async def close(self) -> None:
        pass
