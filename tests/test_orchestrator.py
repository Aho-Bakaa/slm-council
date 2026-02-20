"""Tests for the Orchestrator brain logic."""

import pytest

from slm_council.models import (
    AgentRole,
    RefinementFeedback,
)
from slm_council.orchestrator.brain import Orchestrator


@pytest.fixture
def orchestrator() -> Orchestrator:
    return Orchestrator(endpoint="http://localhost:9999/v1", model="test-model")


class TestExtractJson:
    def test_clean_json(self, orchestrator: Orchestrator) -> None:
        raw = '{"verdict": "APPROVE", "summary": "Looks good"}'
        result = orchestrator._extract_json(raw)
        assert result["verdict"] == "APPROVE"

    def test_fenced_json(self, orchestrator: Orchestrator) -> None:
        raw = "```json\n{\"verdict\": \"REFINE\"}\n```"
        result = orchestrator._extract_json(raw)
        assert result["verdict"] == "REFINE"


class TestBuildRefinementFeedback:
    def test_single_feedback(self, orchestrator: Orchestrator) -> None:
        synthesis = {
            "verdict": "REFINE",
            "refinement_feedback": [
                {
                    "target_agent": "generator",
                    "issues": ["Missing error handling"],
                    "instructions": "Add try/except blocks",
                }
            ],
        }
        fbs = orchestrator.build_refinement_feedback(synthesis, pass_number=1)
        assert len(fbs) == 1
        assert fbs[0].source_agent == AgentRole.GENERATOR
        assert "error handling" in fbs[0].issues[0]

    def test_multiple_feedback(self, orchestrator: Orchestrator) -> None:
        synthesis = {
            "verdict": "REFINE",
            "refinement_feedback": [
                {
                    "target_agent": "generator",
                    "issues": ["Bug A"],
                    "instructions": "Fix A",
                },
                {
                    "target_agent": "researcher",
                    "issues": ["Missing docs"],
                    "instructions": "Add references",
                },
            ],
        }
        fbs = orchestrator.build_refinement_feedback(synthesis, pass_number=2)
        assert len(fbs) == 2
        agents = {fb.source_agent for fb in fbs}
        assert AgentRole.GENERATOR in agents
        assert AgentRole.RESEARCHER in agents

    def test_unknown_agent_skipped(self, orchestrator: Orchestrator) -> None:
        synthesis = {
            "refinement_feedback": [
                {
                    "target_agent": "unknown_agent",
                    "issues": ["X"],
                    "instructions": "Y",
                }
            ],
        }
        fbs = orchestrator.build_refinement_feedback(synthesis, pass_number=1)
        assert len(fbs) == 0
