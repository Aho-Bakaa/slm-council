"""Tests for agent base class utilities."""

import pytest

from slm_council.agents.base import BaseAgent
from slm_council.models import AgentRole


class _DummyAgent(BaseAgent):
    """Concrete subclass for testing abstract base."""

    role = AgentRole.RESEARCHER
    system_prompt = "You are a test agent."

    def build_prompt(self, task_instruction: str, **ctx) -> str:  # type: ignore[override]
        return task_instruction

    def parse_output(self, raw: str):  # type: ignore[override]
        return self._extract_json(raw)


@pytest.fixture
def agent() -> _DummyAgent:
    return _DummyAgent(endpoint="http://localhost:9999/v1", model="test-model")


class TestExtractJson:
    def test_plain_json(self, agent: _DummyAgent) -> None:
        raw = '{"key": "value", "num": 42}'
        result = agent.parse_output(raw)
        assert result == {"key": "value", "num": 42}

    def test_json_in_markdown_fence(self, agent: _DummyAgent) -> None:
        raw = '```json\n{"key": "value"}\n```'
        result = agent.parse_output(raw)
        assert result == {"key": "value"}

    def test_json_embedded_in_text(self, agent: _DummyAgent) -> None:
        raw = 'Here is the result:\n{"answer": 42}\nDone.'
        result = agent.parse_output(raw)
        assert result == {"answer": 42}

    def test_invalid_json_raises(self, agent: _DummyAgent) -> None:
        with pytest.raises(ValueError, match="Could not extract JSON"):
            agent.parse_output("no json here at all")

    def test_nested_json(self, agent: _DummyAgent) -> None:
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        result = agent.parse_output(raw)
        assert result["outer"]["inner"] == [1, 2, 3]
