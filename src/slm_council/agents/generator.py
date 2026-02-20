"""Code Generator agent – Qwen3-Coder 4B.

Consumes the Tech Manifest and writes clean, type-hinted,
production-quality code files.
"""

from __future__ import annotations

import json
from typing import Any

from slm_council.agents.base import BaseAgent
from slm_council.agents.registry import registry
from slm_council.models import AgentRole, CodeFile, GeneratedCode
from slm_council.utils.prompts import GENERATOR_SYSTEM, GENERATOR_TASK


@registry.register(AgentRole.GENERATOR)
class CodeGeneratorAgent(BaseAgent):
    role = AgentRole.GENERATOR
    system_prompt = GENERATOR_SYSTEM

    def build_prompt(self, task_instruction: str, **ctx: Any) -> str:
        tech_manifest = ctx.get("tech_manifest", "{}")
        if not isinstance(tech_manifest, str):
            tech_manifest = json.dumps(tech_manifest, indent=2)

        refinement_context = ""
        if feedback := ctx.get("refinement_feedback"):
            refinement_context = (
                f"**Refinement feedback from Orchestrator (pass {ctx.get('pass_number', '?')}):**\n"
                f"{feedback}\n\n"
                "Fix the issues listed above and regenerate the code."
            )

        architecture_plan = ctx.get("architecture_plan", "N/A")
        if not isinstance(architecture_plan, str):
            architecture_plan = json.dumps(architecture_plan, indent=2)

        return GENERATOR_TASK.format(
            instruction=task_instruction,
            tech_manifest=tech_manifest,
            architecture_plan=architecture_plan,
            refinement_context=refinement_context,
        )

    def parse_output(self, raw: str) -> GeneratedCode:
        try:
            data = self._extract_json(raw)
        except Exception:
            # Safe fallback: return partial result so pipeline can continue
            return GeneratedCode(
                files=[],
                explanation="Generator returned non-JSON or malformed JSON output.",
                assumptions=[f"Raw output preview: {raw[:500]}..." if len(raw) > 500 else raw],
            )
        files = [CodeFile(**f) for f in data.get("files", [])]
        return GeneratedCode(
            files=files,
            explanation=data.get("explanation", ""),
            assumptions=data.get("assumptions", []),
        )
