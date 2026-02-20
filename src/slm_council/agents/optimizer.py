"""Performance Optimizer agent.

Analyses code for algorithmic complexity, memory usage patterns,
and performance bottlenecks.  Suggests concrete improvements.
"""

from __future__ import annotations

import json
from typing import Any

from slm_council.agents.base import BaseAgent
from slm_council.agents.registry import registry
from slm_council.models import AgentRole, Bottleneck, OptimizationReport, Verdict
from slm_council.utils.prompts import OPTIMIZER_SYSTEM, OPTIMIZER_TASK


@registry.register(AgentRole.OPTIMIZER)
class OptimizerAgent(BaseAgent):
    role = AgentRole.OPTIMIZER
    system_prompt = OPTIMIZER_SYSTEM

    def build_prompt(self, task_instruction: str, **ctx: Any) -> str:
        code = ctx.get("code", "")
        if not isinstance(code, str):
            code = json.dumps(code, indent=2)
        tech_manifest = ctx.get("tech_manifest", "{}")
        if not isinstance(tech_manifest, str):
            tech_manifest = json.dumps(tech_manifest, indent=2)

        refinement_context = ""
        if feedback := ctx.get("refinement_feedback"):
            refinement_context = (
                f"**Refinement feedback (pass {ctx.get('pass_number', '?')}):**\n"
                f"{feedback}\n\n"
                "Re-analyse the code for performance with this feedback."
            )

        return OPTIMIZER_TASK.format(
            instruction=task_instruction,
            code=code,
            tech_manifest=tech_manifest,
            refinement_context=refinement_context,
        )

    def parse_output(self, raw: str) -> OptimizationReport:
        try:
            data = self._extract_json(raw)
        except Exception:
            return OptimizationReport(
                verdict=Verdict.PARTIAL,
                complexity_analysis="",
                bottlenecks=[],
                improvements=[],
                memory_notes="",
                overall_assessment="Optimizer returned malformed output; fallback report.",
            )

        verdict_value = str(data.get("verdict", "partial")).lower()
        if verdict_value not in {"pass", "fail", "partial"}:
            verdict_value = "partial"

        bottlenecks: list[Bottleneck] = []
        for b in data.get("bottlenecks", []):
            try:
                bottlenecks.append(Bottleneck(**b))
            except Exception:
                bottlenecks.append(
                    Bottleneck(description=str(b))
                )

        return OptimizationReport(
            verdict=Verdict(verdict_value),
            complexity_analysis=data.get("complexity_analysis", ""),
            bottlenecks=bottlenecks,
            improvements=data.get("improvements", []),
            memory_notes=data.get("memory_notes", ""),
            overall_assessment=data.get("overall_assessment", ""),
        )
