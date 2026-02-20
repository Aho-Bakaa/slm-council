"""Planner / Architect agent.

Designs the solution architecture — components, file layout, class
hierarchy, data flow — BEFORE any code is written.  The Code Generator
follows this plan.
"""

from __future__ import annotations

import json
from typing import Any

from slm_council.agents.base import BaseAgent
from slm_council.agents.registry import registry
from slm_council.models import AgentRole, ArchitecturePlan, ComponentSpec
from slm_council.utils.prompts import PLANNER_SYSTEM, PLANNER_TASK


@registry.register(AgentRole.PLANNER)
class PlannerAgent(BaseAgent):
    role = AgentRole.PLANNER
    system_prompt = PLANNER_SYSTEM

    def build_prompt(self, task_instruction: str, **ctx: Any) -> str:
        tech_manifest = ctx.get("tech_manifest", "{}")
        if not isinstance(tech_manifest, str):
            tech_manifest = json.dumps(tech_manifest, indent=2)

        refinement_context = ""
        if feedback := ctx.get("refinement_feedback"):
            refinement_context = (
                f"**Refinement feedback (pass {ctx.get('pass_number', '?')}):**\n"
                f"{feedback}\n\n"
                "Revise the architecture plan to address these issues."
            )

        return PLANNER_TASK.format(
            instruction=task_instruction,
            tech_manifest=tech_manifest,
            refinement_context=refinement_context,
        )

    def parse_output(self, raw: str) -> ArchitecturePlan:
        try:
            data = self._extract_json(raw)
        except Exception:
            return ArchitecturePlan(
                summary="Planner output was malformed; using fallback plan.",
                components=[],
                file_layout=[],
                class_hierarchy="",
                api_design="",
                data_flow="",
                design_decisions=["Planner parse fallback applied"],
            )

        components: list[ComponentSpec] = []
        for c in data.get("components", []):
            try:
                components.append(ComponentSpec(**c))
            except Exception:
                continue

        return ArchitecturePlan(
            summary=data.get("summary", ""),
            components=components,
            file_layout=data.get("file_layout", []),
            class_hierarchy=data.get("class_hierarchy", ""),
            api_design=data.get("api_design", ""),
            data_flow=data.get("data_flow", ""),
            design_decisions=data.get("design_decisions", []),
        )
