"""Abstract base classes for the Knowledge Provider system.

A KnowledgeProvider gives agents access to contextual knowledge:
- Code patterns from the user's codebase
- Documentation snippets
- Previous solutions to similar problems
- Style preferences

Future implementations:
- ChromaDB / LanceDB vector store
- sentence-transformers embeddings
- Local codebase indexing
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeChunk:
    """A single piece of retrieved knowledge."""

    content: str
    source: str = ""  # file path, URL, or description
    relevance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeProvider(ABC):
    """Abstract base for knowledge retrieval systems.

    Implementations should handle:
    - Indexing: ingest code files, docs, or other sources
    - Retrieval: find relevant knowledge for a given query
    - Lifecycle: initialize and cleanup resources
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Set up the knowledge store (create indices, load embeddings, etc.)."""
        ...

    @abstractmethod
    async def ingest(self, content: str, source: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a piece of content to the knowledge store."""
        ...

    @abstractmethod
    async def query(self, text: str, top_k: int = 5) -> list[KnowledgeChunk]:
        """Retrieve the most relevant knowledge chunks for the given text."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
        ...


class NullKnowledgeProvider(KnowledgeProvider):
    """No-op implementation used when no knowledge store is configured."""

    async def initialize(self) -> None:
        pass

    async def ingest(self, content: str, source: str, metadata: dict[str, Any] | None = None) -> None:
        pass

    async def query(self, text: str, top_k: int = 5) -> list[KnowledgeChunk]:
        return []

    async def close(self) -> None:
        pass
