"""Tester agent – Phi-4-mini.

Generates rigorous unit tests, identifies edge cases, and provides
a Pass/Fail verdict to the Orchestrator.
"""

from __future__ import annotations

import json
from typing import Any

from slm_council.agents.base import BaseAgent
from slm_council.agents.registry import registry
from slm_council.models import (
    AgentRole,
    TestCase,
    TestReport,
    Verdict,
)
from slm_council.utils.prompts import TESTER_SYSTEM, TESTER_TASK


@registry.register(AgentRole.TESTER)
class TesterAgent(BaseAgent):
    role = AgentRole.TESTER
    system_prompt = TESTER_SYSTEM

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
                "Revise the tests and re-evaluate with this feedback."
            )

        return TESTER_TASK.format(
            instruction=task_instruction,
            code=code,
            tech_manifest=tech_manifest,
            refinement_context=refinement_context,
        )

    def parse_output(self, raw: str) -> TestReport:
        try:
            data = self._extract_json(raw)
        except Exception:
            return TestReport(
                verdict=Verdict.PARTIAL,
                test_cases=[],
                edge_cases_identified=[],
                coverage_notes="Tester returned non-JSON or malformed JSON output.",
                pass_count=0,
                fail_count=0,
                failure_details=[
                    "Unable to parse tester output. Returning safe partial report.",
                    (raw[:500] + "...") if len(raw) > 500 else raw,
                ],
            )

        verdict_value = str(data.get("verdict", "partial")).lower()
        if verdict_value not in {"pass", "fail", "partial"}:
            verdict_value = "partial"

        cases: list[TestCase] = []
        for tc in data.get("test_cases", []):
            try:
                cases.append(TestCase(**tc))
            except Exception:
                continue

        failure_details = data.get("failure_details", [])
        if not isinstance(failure_details, list):
            failure_details = [str(failure_details)]

        edge_cases_identified = data.get("edge_cases_identified", [])
        if not isinstance(edge_cases_identified, list):
            edge_cases_identified = [str(edge_cases_identified)]

        return TestReport(
            verdict=Verdict(verdict_value),
            test_cases=cases,
            edge_cases_identified=edge_cases_identified,
            coverage_notes=str(data.get("coverage_notes", "")),
            pass_count=int(data.get("pass_count", 0) or 0),
            fail_count=int(data.get("fail_count", 0) or 0),
            failure_details=[str(item) for item in failure_details],
        )
