"""Refactorer agent.

Takes existing generated code and restructures it for clean architecture,
DRY, SOLID, and better readability.  Preserves ALL behaviour.
"""

from __future__ import annotations

import json
from typing import Any

from slm_council.agents.base import BaseAgent
from slm_council.agents.registry import registry
from slm_council.models import AgentRole, CodeFile, RefactorResult
from slm_council.utils.prompts import REFACTORER_SYSTEM, REFACTORER_TASK


@registry.register(AgentRole.REFACTORER)
class RefactorerAgent(BaseAgent):
    role = AgentRole.REFACTORER
    system_prompt = REFACTORER_SYSTEM

    def build_prompt(self, task_instruction: str, **ctx: Any) -> str:
        code = ctx.get("code", "")
        if not isinstance(code, str):
            code = json.dumps(code, indent=2)

        feedback = ctx.get("review_feedback", "")
        if not isinstance(feedback, str):
            feedback = json.dumps(feedback, indent=2)
        debug_feedback = ctx.get("debug_feedback", "")
        if debug_feedback:
            feedback = f"{feedback}\n\n{debug_feedback}" if feedback else str(debug_feedback)

        refinement_context = ""
        if rf := ctx.get("refinement_feedback"):
            refinement_context = (
                f"**Refinement feedback (pass {ctx.get('pass_number', '?')}):**\n"
                f"{rf}\n\n"
                "Revise the refactoring to address these issues."
            )

        return REFACTORER_TASK.format(
            instruction=task_instruction,
            code=code,
            feedback=feedback or "N/A",
            refinement_context=refinement_context,
        )

    def parse_output(self, raw: str) -> RefactorResult:
        try:
            data = self._extract_json(raw)
        except Exception:
            return RefactorResult(
                files=[],
                changes_made=[],
                patterns_applied=[],
                explanation="Refactorer returned malformed output; fallback report.",
            )

        files: list[CodeFile] = []
        for f in data.get("files", []):
            try:
                files.append(CodeFile(**f))
            except Exception:
                continue

        return RefactorResult(
            files=files,
            changes_made=data.get("changes_made", []),
            patterns_applied=data.get("patterns_applied", []),
            explanation=data.get("explanation", ""),
        )
