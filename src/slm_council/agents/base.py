"""Abstract base class for all council agents.

Every agent communicates with its backing SLM via an **OpenAI-compatible**
``/chat/completions`` endpoint (the format vLLM exposes).  The base class
handles HTTP transport, retries, timeout, and token-usage tracking so that
concrete agents only need to implement prompt construction and output parsing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from slm_council.config import settings
from slm_council.models import AgentResponse, AgentRole, ParseErrorCategory, TaskStatus
from slm_council.utils.logging import get_logger, truncate_for_log

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────
# Caching infrastructure
# ────────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """A cached agent response with timestamp."""
    response: AgentResponse
    timestamp: float
    request_hash: str


class LRUCache:
    """Simple LRU cache with TTL support for agent responses."""

    def __init__(self, max_size: int = 100, ttl_secs: int = 300) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl_secs = ttl_secs

    def _compute_hash(self, model: str, system: str, user: str) -> str:
        content = f"{model}|{system}|{user}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, model: str, system: str, user: str) -> CacheEntry | None:
        key = self._compute_hash(model, system, user)
        if key not in self._cache:
            return None
        entry = self._cache[key]
        # Check TTL
        if time.time() - entry.timestamp > self._ttl_secs:
            del self._cache[key]
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return entry

    def set(self, model: str, system: str, user: str, response: AgentResponse) -> str:
        key = self._compute_hash(model, system, user)
        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = CacheEntry(
            response=response,
            timestamp=time.time(),
            request_hash=key,
        )
        return key

    def clear(self) -> None:
        self._cache.clear()


# Global cache shared by all agents
_request_cache = LRUCache(max_size=200, ttl_secs=settings.cache_ttl_secs)


class BaseAgent(ABC):
    """Thin async wrapper around an OpenAI-compatible chat endpoint."""

    role: AgentRole
    system_prompt: str  # set by each subclass

    def __init__(self, endpoint: str, model: str, api_key: str = "") -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if "openrouter.ai" in self.endpoint:
            headers["HTTP-Referer"] = "https://github.com/slm-council"
            headers["X-Title"] = "SLM Coding Council"

        self._client = httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=httpx.Timeout(settings.request_timeout_secs, connect=10.0),
            headers=headers or None,
        )

        # Conversation memory: store recent (prompt, response) pairs per session
        # Key: session_id, Value: list of (user_prompt, raw_output) tuples
        self._memory: dict[str, list[tuple[str, str]]] = {}

    # ── public API ───────────────────────────────────────────────────

    async def run(
        self,
        task_instruction: str,
        session_id: str = "",
        use_cache: bool = True,
        **ctx: Any,
    ) -> AgentResponse:
        """Execute the agent's task end-to-end and return a uniform response."""
        t0 = time.perf_counter()
        raw = ""
        error_category: ParseErrorCategory | None = None
        request_hash = ""

        try:
            user_prompt = self.build_prompt(task_instruction, **ctx)

            # Check cache first
            if use_cache and settings.enable_request_cache:
                cached = _request_cache.get(self.model, self.system_prompt, user_prompt)
                if cached is not None:
                    logger.info(
                        "agent.cache_hit",
                        agent=self.role.value,
                        hash=cached.request_hash,
                    )
                    # Return cached response with cache_hit flag
                    cached_resp = cached.response.model_copy()
                    cached_resp.cache_hit = True
                    cached_resp.request_hash = cached.request_hash
                    return cached_resp

            if settings.log_agent_io:
                logger.info(
                    "agent.prompt",
                    agent=self.role.value,
                    model=self.model,
                    endpoint=self.endpoint,
                    prompt=truncate_for_log(user_prompt, settings.log_max_chars),
                )

            # Build messages with memory context
            messages = self._build_messages_with_memory(user_prompt, session_id)

            raw, usage = await self._call_model_with_messages(messages)

            if settings.log_agent_io:
                logger.info(
                    "agent.output",
                    agent=self.role.value,
                    output=truncate_for_log(raw, settings.log_max_chars),
                )

            # Store in memory for future passes
            if session_id:
                self._add_to_memory(session_id, user_prompt, raw)

            parsed = self.parse_output(raw)
            elapsed = time.perf_counter() - t0
            logger.info(
                "agent.completed",
                agent=self.role.value,
                secs=round(elapsed, 2),
                tokens=usage,
            )

            response = AgentResponse(
                agent=self.role,
                status=TaskStatus.COMPLETED,
                raw_output=raw,
                parsed=parsed,
                duration_secs=round(elapsed, 2),
                token_usage=usage,
                request_hash=request_hash,
            )

            # Cache successful response
            if use_cache and settings.enable_request_cache:
                request_hash = _request_cache.set(
                    self.model, self.system_prompt, user_prompt, response
                )
                response.request_hash = request_hash

            return response

        except httpx.ReadTimeout:
            elapsed = time.perf_counter() - t0
            error_category = ParseErrorCategory.TIMEOUT
            logger.error("agent.timeout", agent=self.role.value)
            return AgentResponse(
                agent=self.role,
                status=TaskStatus.FAILED,
                error="Request timed out",
                error_category=error_category,
                duration_secs=round(elapsed, 2),
            )
        except httpx.HTTPStatusError as exc:
            elapsed = time.perf_counter() - t0
            if exc.response.status_code == 429:
                error_category = ParseErrorCategory.RATE_LIMITED
            else:
                error_category = ParseErrorCategory.API_ERROR
            logger.error("agent.http_error", agent=self.role.value, status=exc.response.status_code)
            return AgentResponse(
                agent=self.role,
                status=TaskStatus.FAILED,
                error=str(exc),
                error_category=error_category,
                duration_secs=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            error_category = self._categorize_error(exc, raw)
            error_payload: dict[str, Any] = {
                "agent": self.role.value,
                "error": str(exc),
                "error_category": error_category.value if error_category else "unknown",
            }
            if settings.log_agent_io and raw:
                error_payload["raw_output"] = truncate_for_log(raw, settings.log_max_chars)
            logger.error("agent.failed", **error_payload)
            return AgentResponse(
                agent=self.role,
                status=TaskStatus.FAILED,
                error=str(exc),
                error_category=error_category,
                raw_output=raw,
                duration_secs=round(elapsed, 2),
            )

    def _categorize_error(self, exc: Exception, raw: str) -> ParseErrorCategory:
        """Determine structured error category from exception and output."""
        error_msg = str(exc).lower()
        if "json" in error_msg or "parse" in error_msg:
            if not raw or raw.strip() == "":
                return ParseErrorCategory.EMPTY_RESPONSE
            return ParseErrorCategory.JSON_MALFORMED
        if "schema" in error_msg or "validation" in error_msg:
            return ParseErrorCategory.SCHEMA_MISMATCH
        if "timeout" in error_msg:
            return ParseErrorCategory.TIMEOUT
        if "rate" in error_msg or "429" in error_msg:
            return ParseErrorCategory.RATE_LIMITED
        return ParseErrorCategory.UNKNOWN

    def _build_messages_with_memory(
        self,
        user_prompt: str,
        session_id: str,
    ) -> list[dict[str, str]]:
        """Build message list including conversation history for context."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Add memory from previous passes (sliding window)
        if session_id and session_id in self._memory:
            history = self._memory[session_id][-settings.agent_memory_window:]
            for prev_prompt, prev_output in history:
                # Truncate to avoid token explosion
                truncated_prompt = prev_prompt[:2000] + "..." if len(prev_prompt) > 2000 else prev_prompt
                truncated_output = prev_output[:2000] + "..." if len(prev_output) > 2000 else prev_output
                messages.append({"role": "user", "content": f"[Previous attempt]\n{truncated_prompt}"})
                messages.append({"role": "assistant", "content": truncated_output})

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _add_to_memory(self, session_id: str, prompt: str, output: str) -> None:
        """Store prompt/output pair in session memory."""
        if session_id not in self._memory:
            self._memory[session_id] = []
        self._memory[session_id].append((prompt, output))
        # Keep only last N entries to prevent memory bloat
        max_entries = settings.agent_memory_window * 2
        if len(self._memory[session_id]) > max_entries:
            self._memory[session_id] = self._memory[session_id][-max_entries:]

    def clear_memory(self, session_id: str | None = None) -> None:
        """Clear conversation memory for a session or all sessions."""
        if session_id:
            self._memory.pop(session_id, None)
        else:
            self._memory.clear()

    # ── abstract hooks (implement in subclasses) ─────────────────────

    @abstractmethod
    def build_prompt(self, task_instruction: str, **ctx: Any) -> str:
        """Construct the user-message content for the chat completion."""
        ...

    @abstractmethod
    def parse_output(self, raw: str) -> Any:
        """Parse the raw LLM text into a typed Pydantic model."""
        ...

    # ── internal transport ───────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
    )
    async def _call_model_with_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        """Send a chat-completion request with explicit messages and return (text, usage)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4096,
        }
        resp = await self._client.post("/chat/completions", json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = resp.text[:1000]
            except Exception:
                detail = "<no response body>"
            raise httpx.HTTPStatusError(
                f"{exc}. body={detail}",
                request=exc.request,
                response=exc.response,
            ) from exc
        body = resp.json()

        text: str = body["choices"][0]["message"]["content"]
        usage: dict[str, Any] = body.get("usage", {})
        return text, usage

    async def _call_model(self, user_content: str) -> tuple[str, dict[str, Any]]:
        """Legacy method: Send a chat-completion request and return (text, usage)."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return await self._call_model_with_messages(messages)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Best-effort extraction of a JSON object from LLM output.

        Handles common wrapping: ```json ... ``` or plain text.
        """
        # Strip markdown fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # remove first and last fence lines
            lines = cleaned.splitlines()
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        # Handle fenced JSON that appears after headings/prose
        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if fence_match:
            fenced_snippet = fence_match.group(1)
            try:
                return json.loads(fenced_snippet)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                try:
                    literal = ast.literal_eval(fenced_snippet)
                    if isinstance(literal, dict):
                        return literal
                except (SyntaxError, ValueError):
                    pass

        # Try parsing directly
        try:
            return json.loads(cleaned)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Fallback for Python-style dict outputs (single quotes, True/False)
        try:
            literal = ast.literal_eval(cleaned)
            if isinstance(literal, dict):
                return literal
        except (SyntaxError, ValueError):
            pass

        # Fallback: find first { ... last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            snippet = cleaned[start : end + 1]
            try:
                return json.loads(snippet)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                try:
                    literal = ast.literal_eval(snippet)
                    if isinstance(literal, dict):
                        return literal
                except (SyntaxError, ValueError):
                    pass

        raise ValueError(f"Could not extract JSON from agent output:\n{text[:300]}")

    async def close(self) -> None:
        await self._client.aclose()
