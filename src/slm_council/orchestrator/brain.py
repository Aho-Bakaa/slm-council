"""Orchestrator – the central "Brain" of the SLM Coding Council.

Responsibilities:
1. Decompose user queries into a DAG of agent tasks.
2. Synthesise agent reports and decide APPROVE / REFINE.
3. Build refinement DAGs for subsequent passes.
4. Construct the final CouncilResult.
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from slm_council.config import settings
from slm_council.models import (
    AgentResponse,
    AgentRole,
    ArchitecturePlan,
    CouncilResult,
    CouncilSession,
    DebugReport,
    GeneratedCode,
    OptimizationReport,
    RefactorResult,
    RefinementFeedback,
    ReviewReport,
    TechManifest,
    TestReport,
    UserRequest,
    Verdict,
)
from slm_council.orchestrator.dag import DAGTask, DAGTaskResult
from slm_council.utils.logging import get_logger
from slm_council.utils.prompts import (
    ORCHESTRATOR_DECOMPOSE,
    ORCHESTRATOR_SYNTHESISE,
    ORCHESTRATOR_SYSTEM,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class ComplexityProfile:
    """Agent-budget and pass-limit for a given task complexity."""
    allowed_agents: frozenset[str]
    max_passes: int


COMPLEXITY_PROFILES: dict[str, ComplexityProfile] = {
    "simple": ComplexityProfile(
        allowed_agents=frozenset({"researcher", "generator", "tester"}),
        max_passes=1,
    ),
    "moderate": ComplexityProfile(
        allowed_agents=frozenset(
            {"researcher", "planner", "generator", "reviewer", "debugger", "tester"}
        ),
        max_passes=2,
    ),
    "complex": ComplexityProfile(
        allowed_agents=frozenset(
            {
                "researcher", "planner", "generator", "reviewer",
                "debugger", "tester", "optimizer", "refactorer",
            }
        ),
        max_passes=settings.max_refinement_passes,
    ),
}

_DEFAULT_PROFILE_KEY = "moderate"


@dataclass
class DecomposeResult:
    """Bundle returned by Orchestrator.decompose()."""
    tasks: list[DAGTask]
    complexity: str  # "simple" | "moderate" | "complex"
    profile: ComplexityProfile = field(init=False)

    def __post_init__(self) -> None:
        self.profile = COMPLEXITY_PROFILES.get(
            self.complexity, COMPLEXITY_PROFILES[_DEFAULT_PROFILE_KEY]
        )


def _truncate_for_log(value: str) -> str:
    if settings.log_max_chars <= 0 or len(value) <= settings.log_max_chars:
        return value
    return f"{value[:settings.log_max_chars]}... <truncated {len(value) - settings.log_max_chars} chars>"


class Orchestrator:
    """The 'One' in the Many-to-One architecture."""

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
    ) -> None:
        self.endpoint = (endpoint or settings.orchestrator_endpoint).rstrip("/")
        self.model = model or settings.orchestrator_model_name

        headers: dict[str, str] = {}
        if settings.orchestrator_api_key:
            headers["Authorization"] = f"Bearer {settings.orchestrator_api_key}"
        if "openrouter.ai" in self.endpoint:
            headers["HTTP-Referer"] = "https://github.com/slm-council"
            headers["X-Title"] = "SLM Coding Council"

        self._client = httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=httpx.Timeout(settings.request_timeout_secs, connect=10.0),
            headers=headers or None,
        )


    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
    )
    async def _call_llm(self, user_content: str) -> str:
        if settings.log_agent_io:
            logger.info(
                "orchestrator.prompt",
                model=self.model,
                endpoint=self.endpoint,
                prompt=_truncate_for_log(user_content),
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM.format(
                    max_passes=settings.max_refinement_passes,
                )},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }
        resp = await self._client.post("/chat/completions", json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = resp.text[:1000]
            except Exception:
                detail = "<no response body>"
            raise httpx.HTTPStatusError(
                f"{exc}. body={detail}",
                request=exc.request,
                response=exc.response,
            ) from exc
        output = resp.json()["choices"][0]["message"]["content"]
        if settings.log_agent_io:
            logger.info(
                "orchestrator.output",
                output=_truncate_for_log(output),
            )
        return output


    _CRITICAL_AGENTS: frozenset[str] = frozenset({"generator"})

    async def decompose(self, request: UserRequest) -> DecomposeResult:
        """Ask the LLM to break the request into a DAG of agent tasks.

        Returns a *DecomposeResult* containing the task list **and** the
        complexity classification used for adaptive agent budgeting.

        Includes retry-with-simplified-prompt on parse failure and a
        hardcoded minimal fallback DAG as a last resort.
        """
        prompts = [
            ORCHESTRATOR_DECOMPOSE.format(
                user_query=request.query,
                language=request.language,
            ),
            None,
        ]

        data: dict[str, Any] | None = None
        for attempt in range(2):
            prompt = prompts[attempt]
            if prompt is None:
                prompt = self._simplified_decompose_prompt(request)
            try:
                raw = await self._call_llm(prompt)
                data = self._extract_json(raw)
                break
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "orchestrator.decompose_parse_failed",
                    attempt=attempt + 1,
                    error=str(exc),
                )

        if data is None:
            logger.warning("orchestrator.decompose_fallback")
            data = {
                "tasks": [
                    {
                        "id": "t1",
                        "agent": "researcher",
                        "instruction": f"Research technologies and approaches for: {request.query}",
                        "dependencies": [],
                    },
                    {
                        "id": "t2",
                        "agent": "generator",
                        "instruction": f"Implement the solution for: {request.query}",
                        "dependencies": ["t1"],
                    },
                    {
                        "id": "t3",
                        "agent": "tester",
                        "instruction": f"Write unit tests for the generated code for: {request.query}",
                        "dependencies": ["t2"],
                    },
                ],
                "complexity": "simple",
            }

        complexity = data.get("complexity", "unknown").lower().strip()

        user_override = request.context.get("complexity")
        if isinstance(user_override, str) and user_override.lower().strip() in COMPLEXITY_PROFILES:
            logger.info(
                "orchestrator.complexity_override",
                llm_said=complexity,
                user_override=user_override.lower().strip(),
            )
            complexity = user_override.lower().strip()

        tasks = self._parse_dag_tasks(data)
        self._assign_error_policies(tasks)
        tasks = self._filter_by_complexity(tasks, complexity)

        result = DecomposeResult(tasks=tasks, complexity=complexity)

        logger.info(
            "orchestrator.decomposed",
            task_count=len(tasks),
            complexity=complexity,
            profile_max_passes=result.profile.max_passes,
            allowed_agents=sorted(result.profile.allowed_agents),
            agents=[t.agent for t in tasks],
        )
        return result


    def _parse_dag_tasks(self, data: dict[str, Any]) -> list[DAGTask]:
        """Extract valid DAGTasks from orchestrator JSON, filtering bad roles."""
        tasks: list[DAGTask] = []
        valid_roles = {r.value for r in AgentRole if r != AgentRole.ORCHESTRATOR}

        for t in data.get("tasks", []):
            agent_name = t.get("agent", "").lower().strip()
            if agent_name not in valid_roles:
                logger.warning("orchestrator.unknown_agent", agent=agent_name)
                continue
            task_id = t.get("id", f"t{len(tasks) + 1}")
            tasks.append(
                DAGTask(
                    id=task_id,
                    agent=agent_name,
                    instruction=t.get("instruction", ""),
                    dependencies=t.get("dependencies", []),
                    extra_context=t.get("extra_context", []),
                )
            )

        if not any(t.agent == "generator" for t in tasks):
            tasks.append(
                DAGTask(
                    id=f"t{len(tasks) + 1}",
                    agent="generator",
                    instruction="Generate the implementation code based on available context.",
                    dependencies=[t.id for t in tasks if t.agent == "researcher"],
                )
            )
        return tasks

    def _assign_error_policies(self, tasks: list[DAGTask]) -> None:
        """Set error_policy on each task based on agent criticality."""
        for task in tasks:
            task.error_policy = (
                "abort" if task.agent in self._CRITICAL_AGENTS else "skip"
            )

    @staticmethod
    def _filter_by_complexity(
        tasks: list[DAGTask], complexity: str
    ) -> list[DAGTask]:
        """Enforce the complexity profile's agent budget.

        Removes tasks whose agent is not in the profile's allowed set,
        then fixes dangling dependency references.
        """
        profile = COMPLEXITY_PROFILES.get(
            complexity, COMPLEXITY_PROFILES[_DEFAULT_PROFILE_KEY]
        )
        allowed = profile.allowed_agents

        kept: list[DAGTask] = []
        removed_labels: list[str] = []
        for t in tasks:
            if t.agent in allowed:
                kept.append(t)
            else:
                removed_labels.append(f"{t.id}:{t.agent}")

        if removed_labels:
            logger.info(
                "orchestrator.complexity_filter",
                complexity=complexity,
                removed=removed_labels,
                remaining=[f"{t.id}:{t.agent}" for t in kept],
            )

        kept_ids = {t.id for t in kept}
        for t in kept:
            t.dependencies = [d for d in t.dependencies if d in kept_ids]

        return kept

    def _simplified_decompose_prompt(self, request: UserRequest) -> str:
        """A bare-bones decomposition prompt used as retry fallback.

        Issue #5: Includes complexity guidance so even the retry attempt
        produces right-sized DAGs.
        """
        return (
            "Break this coding task into agent sub-tasks.\n"
            "Available agents: researcher, planner, generator, reviewer, "
            "debugger, tester, optimizer, refactorer.\n\n"
            "Complexity guide:\n"
            "- simple (utility funcs, single algorithms): researcher + generator + tester ONLY\n"
            "- moderate (multi-function, API endpoints): up to 6 agents (skip optimizer, refactorer)\n"
            "- complex (full systems, auth/db/async): any agents that add value\n\n"
            f"Task: {request.query}\n"
            f"Language: {request.language}\n\n"
            'Respond ONLY with JSON:\n'
            '{"complexity": "simple|moderate|complex", '
            '"tasks": [{"id": "t1", "agent": "researcher", "instruction": "...", "dependencies": []}, '
            '{"id": "t2", "agent": "generator", "instruction": "...", "dependencies": ["t1"]}]}'
        )


    async def synthesise(
        self,
        session: CouncilSession,
        task_results: dict[str, DAGTaskResult],
        prior_syntheses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate all agent reports from the DAG and decide APPROVE or REFINE.

        Gap D: *prior_syntheses* carries the orchestrator's own prior
        verdicts / summaries so it remembers what it already tried.
        """

        report_parts: list[str] = []
        report_chars = settings.synthesis_max_report_chars
        code_chars = settings.synthesis_max_code_chars

        for task_id, tr in task_results.items():
            if tr.error:
                report_parts.append(
                    f"### [{task_id}] {tr.agent} — ERROR\n{tr.error}"
                )
                continue

            if tr.parsed is None:
                report_parts.append(
                    f"### [{task_id}] {tr.agent}\nNo structured output."
                )
                continue

            agent = tr.agent.lower()
            if agent == "generator" and isinstance(tr.parsed, GeneratedCode):
                report_parts.append(
                    f"### [{task_id}] generator\n"
                    + self._summarize_code(tr.parsed, code_chars)
                )
            elif hasattr(tr.parsed, "model_dump"):
                text = json.dumps(tr.parsed.model_dump(), indent=2, default=str)
                if len(text) > report_chars:
                    text = text[:report_chars] + f"\n... <truncated>"
                report_parts.append(f"### [{task_id}] {agent}\n{text}")
            else:
                text = str(tr.parsed)[:report_chars]
                report_parts.append(f"### [{task_id}] {agent}\n{text}")

        agent_reports = "\n\n".join(report_parts) or "No agent reports available."

        _SYNTH_CAP = 16000
        if len(agent_reports) > _SYNTH_CAP:
            agent_reports = agent_reports[:_SYNTH_CAP] + "\n[\u2026truncated]"

        history_text = ""
        if prior_syntheses:
            history_parts = []
            for ps in prior_syntheses:
                history_parts.append(
                    f"- Pass {ps['pass']}: verdict={ps['verdict']}, "
                    f"summary={ps['summary'][:200]}"
                )
            history_text = (
                "\n\n**Your prior decisions (for context — do NOT repeat failed approaches):**\n"
                + "\n".join(history_parts)
            )

        prompt = ORCHESTRATOR_SYNTHESISE.format(
            user_query=session.request.query,
            agent_reports=agent_reports,
            current_pass=session.current_pass,
            max_passes=settings.max_refinement_passes,
            prior_history=history_text,
        )
        raw = await self._call_llm(prompt)
        try:
            return self._extract_json(raw)
        except Exception as exc:
            logger.warning(
                "orchestrator.synthese_parse_fallback",
                error=str(exc),
                raw_preview=raw[:400],
            )
            return {
                "verdict": "REFINE",
                "summary": "Orchestrator returned malformed JSON; applying safe fallback refinement.",
                "refinement_tasks": [
                    {
                        "id": "fix1",
                        "agent": "generator",
                        "instruction": (
                            "Previous synthesis response was malformed. "
                            "Regenerate complete implementation with robust edge-case handling."
                        ),
                        "dependencies": [],
                    }
                ],
            }

    @staticmethod
    def _summarize_code(code: GeneratedCode, max_chars: int) -> str:
        """Smart truncation for code — preserve structure, truncate content."""
        if not code.files:
            return "No files generated"
        parts: list[str] = []
        remaining = max_chars
        for f in code.files:
            header = f"=== {f.filename} ({f.language}) ===\n"
            if remaining <= len(header):
                parts.append(f"... and {len(code.files) - len(parts)} more files")
                break
            content = f.content[: remaining - len(header)]
            if len(f.content) > remaining - len(header):
                content += f"\n... <truncated {len(f.content) - (remaining - len(header))} chars>"
            parts.append(header + content)
            remaining -= len(header) + len(content)
        return "\n\n".join(parts)

    def build_refinement_tasks(
        self,
        synthesis: dict[str, Any],
        complexity: str = _DEFAULT_PROFILE_KEY,
    ) -> list[DAGTask]:
        """Convert the LLM's synthesis refinement into DAGTasks for re-execution.

        Issue #1: the *complexity* parameter ensures refinement tasks are
        filtered through the same complexity profile as the initial DAG,
        preventing synthesis from sneaking in agents outside the budget.
        """
        tasks: list[DAGTask] = []
        valid_roles = {r.value for r in AgentRole if r != AgentRole.ORCHESTRATOR}

        for t in synthesis.get("refinement_tasks", []):
            agent_name = t.get("agent", "").lower().strip()
            if agent_name not in valid_roles:
                continue
            tasks.append(
                DAGTask(
                    id=t.get("id", f"r{len(tasks) + 1}"),
                    agent=agent_name,
                    instruction=t.get("instruction", ""),
                    dependencies=t.get("dependencies", []),
                    extra_context=t.get("extra_context", []),
                )
            )

        if not tasks:
            tasks.append(
                DAGTask(
                    id="r1",
                    agent="generator",
                    instruction="Regenerate the code addressing all previous feedback.",
                    dependencies=[],
                )
            )

        self._assign_error_policies(tasks)

        tasks = self._filter_by_complexity(tasks, complexity)

        return tasks


    def build_result(
        self,
        session: CouncilSession,
        task_results: dict[str, DAGTaskResult],
        execution_order: list[list[str]],
        summary: str,
        total_duration: float,
    ) -> CouncilResult:
        """Build the final CouncilResult from all DAG task outputs."""
        tech_manifest: TechManifest | None = None
        architecture_plan: ArchitecturePlan | None = None
        generated_code: GeneratedCode | None = None
        review_report: ReviewReport | None = None
        debug_report: DebugReport | None = None
        test_report: TestReport | None = None
        optimization_report: OptimizationReport | None = None
        refactored_code: RefactorResult | None = None
        agents_used: list[str] = []

        for tr in task_results.values():
            if tr.agent not in agents_used:
                agents_used.append(tr.agent)

            parsed = tr.parsed
            if parsed is None:
                continue

            if isinstance(parsed, TechManifest):
                tech_manifest = parsed
            elif isinstance(parsed, ArchitecturePlan):
                architecture_plan = parsed
            elif isinstance(parsed, GeneratedCode):
                generated_code = parsed
            elif isinstance(parsed, ReviewReport):
                review_report = parsed
            elif isinstance(parsed, DebugReport):
                debug_report = parsed
            elif isinstance(parsed, TestReport):
                test_report = parsed
            elif isinstance(parsed, OptimizationReport):
                optimization_report = parsed
            elif isinstance(parsed, RefactorResult):
                if parsed.files:
                    refactored_code = parsed
                    generated_code = GeneratedCode(
                        files=parsed.files,
                        explanation=f"Refactored: {parsed.explanation}",
                        assumptions=[],
                    )

        has_code = bool(generated_code and getattr(generated_code, "files", None))
        verdicts: list[Verdict] = []
        if debug_report:
            verdicts.append(debug_report.verdict)
        if test_report:
            verdicts.append(test_report.verdict)
        if review_report:
            verdicts.append(review_report.verdict)
        if optimization_report:
            verdicts.append(optimization_report.verdict)

        if not has_code:
            status = Verdict.FAIL
        elif any(v == Verdict.FAIL for v in verdicts):
            status = Verdict.FAIL
        elif verdicts and all(v == Verdict.PASS for v in verdicts):
            status = Verdict.PASS
        elif verdicts:
            status = Verdict.PARTIAL
        else:
            status = Verdict.PARTIAL

        return CouncilResult(
            session_id=session.id,
            status=status,
            code=generated_code,
            debug_report=debug_report,
            test_report=test_report,
            tech_manifest=tech_manifest,
            architecture_plan=architecture_plan,
            review_report=review_report,
            optimization_report=optimization_report,
            refactored_code=refactored_code,
            refinement_passes=session.current_pass,
            total_duration_secs=round(total_duration, 2),
            summary=summary,
            agents_used=agents_used,
            dag_execution_order=execution_order,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()

        def try_load(candidate: str) -> dict[str, Any] | None:
            candidate = candidate.strip()
            if not candidate:
                return None
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    return loaded
            except json.JSONDecodeError:
                pass
            try:
                loaded = ast.literal_eval(candidate)
                if isinstance(loaded, dict):
                    return loaded
            except (SyntaxError, ValueError):
                pass
            return None

        fence_iter = re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        for match in fence_iter:
            parsed = try_load(match.group(1))
            if parsed is not None:
                return parsed

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        parsed_full = try_load(cleaned)
        if parsed_full is not None:
            return parsed_full

        decoder = json.JSONDecoder()
        first_brace = cleaned.find("{")
        if first_brace != -1:
            try:
                obj, _ = decoder.raw_decode(cleaned[first_brace:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed_snippet = try_load(cleaned[start : end + 1])
            if parsed_snippet is not None:
                return parsed_snippet

        raise ValueError(f"Could not parse orchestrator JSON:\n{text[:300]}")

    async def close(self) -> None:
        await self._client.aclose()
