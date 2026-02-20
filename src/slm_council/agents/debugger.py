"""Debugger agent – DeepSeek-R1-Distill-Qwen-7B.

Uses Chain-of-Thought reasoning to trace code logic, find bugs,
race conditions, and integration issues.
"""

from __future__ import annotations

import json
from typing import Any

from slm_council.agents.base import BaseAgent
from slm_council.agents.registry import registry
from slm_council.models import (
    AgentRole,
    BugReport,
    DebugReport,
    Severity,
    Verdict,
)
from slm_council.utils.prompts import DEBUGGER_SYSTEM, DEBUGGER_TASK


@registry.register(AgentRole.DEBUGGER)
class DebuggerAgent(BaseAgent):
    role = AgentRole.DEBUGGER
    system_prompt = DEBUGGER_SYSTEM

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
                f"**Refinement feedback from Orchestrator (pass {ctx.get('pass_number', '?')}):**\n"
                f"{feedback}\n\n"
                "Re-analyse the updated code with this feedback in mind."
            )

        return DEBUGGER_TASK.format(
            instruction=task_instruction,
            code=code,
            tech_manifest=tech_manifest,
            refinement_context=refinement_context,
        )

    def parse_output(self, raw: str) -> DebugReport:
        try:
            data = self._extract_json(raw)
        except Exception:
            return DebugReport(
                verdict=Verdict.PARTIAL,
                reasoning_trace="Debugger returned non-JSON or malformed JSON output.",
                bugs=[
                    BugReport(
                        file="",
                        line_range="",
                        severity=Severity.WARNING,
                        category="parser",
                        description="Unable to parse debugger output. Returning safe partial report.",
                        suggested_fix=(raw[:500] + "...") if len(raw) > 500 else raw,
                    )
                ],
                overall_assessment="Partial: debugger output could not be parsed.",
            )

        reasoning_value = data.get("reasoning_trace", "")
        if isinstance(reasoning_value, list):
            reasoning_trace = "\n".join(
                json.dumps(step, ensure_ascii=False) if isinstance(step, (dict, list)) else str(step)
                for step in reasoning_value
            )
        else:
            reasoning_trace = str(reasoning_value)

        verdict_value = str(data.get("verdict", "partial")).lower()
        if verdict_value not in {"pass", "fail", "partial"}:
            verdict_value = "partial"

        bugs = []
        for b in data.get("bugs", []):
            severity_value = str(b.get("severity", "warning")).lower()
            if severity_value not in {"info", "warning", "error", "critical"}:
                severity_value = "warning"
            bugs.append(
                BugReport(
                    file=b.get("file", ""),
                    line_range=b.get("line_range", ""),
                    severity=Severity(severity_value),
                    category=b.get("category", ""),
                    description=b.get("description", ""),
                    suggested_fix=b.get("suggested_fix", ""),
                )
            )

        if not bugs and verdict_value != "pass":
            bugs.append(
                BugReport(
                    file="",
                    line_range="",
                    severity=Severity.WARNING,
                    category="analysis",
                    description="Debugger marked non-pass but returned no structured bugs.",
                    suggested_fix="Review raw debugger response and enforce output schema.",
                )
            )

        return DebugReport(
            verdict=Verdict(verdict_value),
            reasoning_trace=reasoning_trace,
            bugs=bugs,
            overall_assessment=data.get("overall_assessment", ""),
        )
