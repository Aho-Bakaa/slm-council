"""Tech Researcher agent – Gemma 3 4B-IT.

Scans documentation, frameworks, and API specs to produce a structured
Tech Manifest that downstream agents consume.
"""

from __future__ import annotations

from typing import Any

from slm_council.agents.base import BaseAgent
from slm_council.agents.registry import registry
from slm_council.models import AgentRole, DependencySpec, TechManifest
from slm_council.utils.prompts import RESEARCHER_SYSTEM, RESEARCHER_TASK


@registry.register(AgentRole.RESEARCHER)
class TechResearcherAgent(BaseAgent):
    role = AgentRole.RESEARCHER
    system_prompt = RESEARCHER_SYSTEM

    def build_prompt(self, task_instruction: str, **ctx: Any) -> str:
        language = ctx.get("language", "python")
        refinement_context = ""
        if feedback := ctx.get("refinement_feedback"):
            refinement_context = (
                f"**Refinement feedback from Orchestrator (pass {ctx.get('pass_number', '?')}):**\n"
                f"{feedback}\n\n"
                "Incorporate this feedback and produce an improved Tech Manifest."
            )
        return RESEARCHER_TASK.format(
            instruction=task_instruction,
            language=language,
            refinement_context=refinement_context,
        )

    def parse_output(self, raw: str) -> TechManifest:
        try:
            data = self._extract_json(raw)
        except Exception:
            return TechManifest(
                summary="Research output was malformed; using fallback manifest.",
                architecture_notes="Researcher response could not be parsed into strict JSON.",
                dependencies=[],
                api_contracts=[],
                constraints=["Researcher parse fallback applied"],
                design_patterns=[],
                references=[],
            )

        deps = [DependencySpec(**d) for d in data.get("dependencies", [])]
        return TechManifest(
            summary=data.get("summary", ""),
            architecture_notes=data.get("architecture_notes", ""),
            dependencies=deps,
            api_contracts=data.get("api_contracts", []),
            constraints=data.get("constraints", []),
            design_patterns=data.get("design_patterns", []),
            references=data.get("references", []),
        )
