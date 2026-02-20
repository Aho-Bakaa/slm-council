"""Code Reviewer agent.

Reviews code for style, security, naming conventions, and best practices.
Does NOT fix bugs — that is the Debugger's job.
"""

from __future__ import annotations

import json
from typing import Any

from slm_council.agents.base import BaseAgent
from slm_council.agents.registry import registry
from slm_council.models import AgentRole, ReviewReport, StyleIssue, Verdict
from slm_council.utils.prompts import REVIEWER_SYSTEM, REVIEWER_TASK


@registry.register(AgentRole.REVIEWER)
class ReviewerAgent(BaseAgent):
    role = AgentRole.REVIEWER
    system_prompt = REVIEWER_SYSTEM

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
                "Re-review the code with this feedback in mind."
            )

        return REVIEWER_TASK.format(
            instruction=task_instruction,
            code=code,
            tech_manifest=tech_manifest,
            refinement_context=refinement_context,
        )

    def parse_output(self, raw: str) -> ReviewReport:
        try:
            data = self._extract_json(raw)
        except Exception:
            return ReviewReport(
                verdict=Verdict.PARTIAL,
                style_issues=[],
                security_issues=[],
                best_practice_violations=[],
                suggestions=[],
                overall_assessment="Reviewer returned malformed output; fallback report.",
            )

        verdict_value = str(data.get("verdict", "partial")).lower()
        if verdict_value not in {"pass", "fail", "partial"}:
            verdict_value = "partial"

        def _parse_issues(items: list[dict]) -> list[StyleIssue]:
            issues: list[StyleIssue] = []
            for item in items:
                try:
                    issues.append(StyleIssue(**item))
                except Exception:
                    issues.append(
                        StyleIssue(
                            description=str(item),
                            category="unknown",
                        )
                    )
            return issues

        return ReviewReport(
            verdict=Verdict(verdict_value),
            style_issues=_parse_issues(data.get("style_issues", [])),
            security_issues=_parse_issues(data.get("security_issues", [])),
            best_practice_violations=data.get("best_practice_violations", []),
            suggestions=data.get("suggestions", []),
            overall_assessment=data.get("overall_assessment", ""),
        )
