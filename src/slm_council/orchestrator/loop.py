

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from tenacity import RetryError

from slm_council.agents.base import BaseAgent
from slm_council.agents.registry import registry
from slm_council.config import settings
from slm_council.models import (
    AgentRole,
    CouncilResult,
    CouncilSession,
    DebugReport,
    OptimizationReport,
    ReviewReport,
    TaskStatus,
    TestReport,
    UserRequest,
    Verdict,
)
from slm_council.orchestrator.brain import Orchestrator
from slm_council.orchestrator.brain import DecomposeResult
from slm_council.orchestrator.dag import (
    DAGExecutionResult,
    DAGExecutor,
    DAGTask,
    DAGTaskResult,
    DAGValidationError,
)
from slm_council.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Health-score helper  (used by convergence gate)
# ─────────────────────────────────────────────────────────────────

_VERDICT_SCORE_MAP = {
    Verdict.PASS: 1.0,
    Verdict.PARTIAL: 0.5,
    Verdict.FAIL: 0.0,
}


def _verdict_score(v: Verdict) -> float:
    return _VERDICT_SCORE_MAP.get(v, 0.0)


def _compute_health_score(task_results: dict[str, DAGTaskResult]) -> float:
    """Compute a weighted 0.0–1.0 health score from available agent reports.

    Weights:
      - Test pass rate : 3.0  (most reliable signal)
      - Debug verdict  : 2.0
      - Review verdict : 1.0
      - Optimizer      : 0.5
    Returns 0.0 when no scorable reports are available.
    """
    scores: list[float] = []
    weights: list[float] = []

    for tr in task_results.values():
        if tr.parsed is None or tr.error:
            continue

        if isinstance(tr.parsed, TestReport):
            total = tr.parsed.pass_count + tr.parsed.fail_count
            if total > 0:
                scores.append(tr.parsed.pass_count / total)
                weights.append(3.0)
            scores.append(_verdict_score(tr.parsed.verdict))
            weights.append(1.0)

        elif isinstance(tr.parsed, DebugReport):
            scores.append(_verdict_score(tr.parsed.verdict))
            weights.append(2.0)

        elif isinstance(tr.parsed, ReviewReport):
            scores.append(_verdict_score(tr.parsed.verdict))
            weights.append(1.0)

        elif isinstance(tr.parsed, OptimizationReport):
            scores.append(_verdict_score(tr.parsed.verdict))
            weights.append(0.5)

    if not scores:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def _ensure_agents_loaded() -> None:
    """Import all agent modules so they register with the registry."""
    # Each import triggers the @registry.register decorator at module scope
    import slm_council.agents.researcher  
    import slm_council.agents.planner  
    import slm_council.agents.generator  
    import slm_council.agents.reviewer  
    import slm_council.agents.debugger  
    import slm_council.agents.tester  
    import slm_council.agents.optimizer  
    import slm_council.agents.refactorer  


# Agent config mapping: role name → (endpoint_setting, model_setting, api_key_setting)
_AGENT_CONFIG_MAP: dict[str, tuple[str, str, str]] = {
    "researcher": ("researcher_endpoint", "researcher_model", "researcher_api_key"),
    "planner": ("planner_endpoint", "planner_model", "planner_api_key"),
    "generator": ("generator_endpoint", "generator_model", "generator_api_key"),
    "reviewer": ("reviewer_endpoint", "reviewer_model", "reviewer_api_key"),
    "debugger": ("debugger_endpoint", "debugger_model", "debugger_api_key"),
    "tester": ("tester_endpoint", "tester_model", "tester_api_key"),
    "optimizer": ("optimizer_endpoint", "optimizer_model", "optimizer_api_key"),
    "refactorer": ("refactorer_endpoint", "refactorer_model", "refactorer_api_key"),
}


def _create_agent(role_name: str) -> BaseAgent | None:
    """Create an agent instance from config, or None if unconfigured."""
    cfg = _AGENT_CONFIG_MAP.get(role_name)
    if cfg is None:
        return None

    endpoint_attr, model_attr, key_attr = cfg
    endpoint = getattr(settings, endpoint_attr, "")
    model = getattr(settings, model_attr, "")
    api_key = getattr(settings, key_attr, "")

    if not endpoint or not model:
        logger.warning(
            "agent.unconfigured",
            role=role_name,
            endpoint=endpoint_attr,
            model=model_attr,
        )
        return None

    try:
        role = AgentRole(role_name)
    except ValueError:
        return None

    return registry.create(role, endpoint=endpoint, model=model, api_key=api_key)


class CouncilLoop:
    """Manages a single end-to-end council pipeline execution."""

    def __init__(self) -> None:
        _ensure_agents_loaded()

        self.orchestrator = Orchestrator(
            endpoint=settings.orchestrator_endpoint,
            model=settings.orchestrator_model_name,
        )

        # Create agents dynamically from registry + config
        self.agents: dict[str, BaseAgent] = {}
        for role_name in _AGENT_CONFIG_MAP:
            agent = _create_agent(role_name)
            if agent is not None:
                self.agents[role_name] = agent

        logger.info(
            "council.init",
            available_agents=list(self.agents.keys()),
            registered_roles=[r.value for r in registry.available_roles()],
        )

    # ─────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────

    async def run(self, request: UserRequest) -> CouncilResult:
        """Execute the full council pipeline for a user request."""
        t0 = time.perf_counter()
        session = CouncilSession(request=request)

        max_passes = settings.max_refinement_passes
        context_max = request.context.get("max_iterations")
        if context_max is not None:
            try:
                max_passes = max(1, min(int(context_max), 50))
            except (TypeError, ValueError):
                max_passes = settings.max_refinement_passes

        logger.info("council.start", session_id=session.id, query=request.query[:80])

        # Cumulative results across refinement passes
        all_task_results: dict[str, DAGTaskResult] = {}
        all_execution_order: list[list[str]] = []
        summary = ""
        prev_health: float = -1.0  # convergence gate seed

        try:
            # Phase 1 – Orchestrator decomposes the task into a DAG
            decomposition = await self.orchestrator.decompose(request)
            dag_tasks = decomposition.tasks
            complexity = decomposition.complexity  # persist for refinement passes
            session.status = TaskStatus.IN_PROGRESS

            # ── Adaptive pass limit from complexity profile ─────────
            profile_max = decomposition.profile.max_passes
            max_passes = min(max_passes, profile_max)
            logger.info(
                "council.complexity_routing",
                complexity=complexity,
                profile_max_passes=profile_max,
                effective_max_passes=max_passes,
                agent_count=len(dag_tasks),
                agents=[t.agent for t in dag_tasks],
            )

            # Build shared context from user request
            shared_ctx: dict[str, Any] = {
                "language": request.language,
            }

            # Create the DAG executor
            semaphore = asyncio.Semaphore(settings.max_concurrent_agent_calls)

            # Orchestrator synthesis history (Gap D)
            prior_syntheses: list[dict[str, Any]] = []

            # ── Iteration loop ──────────────────────────────────────
            for pass_num in range(1, max_passes + 1):
                session.current_pass = pass_num

                # ── Gap B: pass number available to every agent ─────
                shared_ctx["pass_number"] = pass_num

                logger.info(
                    "council.pass",
                    pass_number=pass_num,
                    session_id=session.id,
                    task_count=len(dag_tasks),
                )

                # Filter tasks to only those we have agents for
                executable_tasks = self._filter_executable(dag_tasks)
                if not executable_tasks:
                    logger.error("council.no_executable_tasks", pass_number=pass_num)
                    summary = "No executable tasks — all required agents are unconfigured."
                    break

                # Create the DAG executor
                executor = DAGExecutor(
                    agents=self.agents,
                    semaphore=semaphore,
                    session_id=session.id,
                    task_timeout_secs=settings.agent_task_timeout_secs,
                )

                try:
                    dag_result = await executor.execute(
                        executable_tasks, shared_context=shared_ctx
                    )
                except DAGValidationError as exc:
                    logger.error("council.dag_invalid", error=str(exc))
                    summary = f"DAG validation failed: {exc}"
                    break

                # Merge results (later passes override earlier for same role)
                all_task_results.update(dag_result.task_results)
                all_execution_order.extend(dag_result.execution_order)

                if dag_result.errors:
                    logger.warning(
                        "council.dag_errors",
                        errors=dag_result.errors,
                        pass_number=pass_num,
                    )

                # ── Convergence gate ───────────────────────────────
                current_health = _compute_health_score(all_task_results)
                logger.info(
                    "council.health",
                    score=round(current_health, 3),
                    previous=round(prev_health, 3),
                    pass_number=pass_num,
                )
                if pass_num > 1 and current_health <= prev_health:
                    logger.info(
                        "council.convergence_stall",
                        current=round(current_health, 3),
                        previous=round(prev_health, 3),
                        pass_number=pass_num,
                    )
                    summary = (
                        f"Convergence stall detected "
                        f"(health {current_health:.2f} ≤ {prev_health:.2f}). "
                        f"Stopping refinement after pass {pass_num}."
                    )
                    break
                prev_health = current_health

                # Issue #2: skip synthesis when this is the last allowed
                # pass — avoids a wasted LLM call for single-pass (simple)
                # tasks and for the final pass of any tier.
                if pass_num >= max_passes:
                    has_code = self._has_generated_code(all_task_results)
                    if has_code:
                        summary = "Completed in a single pass (simple task)."
                        logger.info("council.single_pass_done", pass_number=pass_num)
                    else:
                        summary = (
                            "Completed all allowed passes but no code was generated."
                        )
                        logger.warning("council.no_code_final_pass", pass_number=pass_num)
                    break

                # Phase 2 – Orchestrator synthesises all results
                # Gap D: pass prior synthesis history so orchestrator
                # knows what it previously decided / asked to fix
                synthesis = await self.orchestrator.synthesise(
                    session, all_task_results, prior_syntheses=prior_syntheses
                )

                verdict_raw = synthesis.get("verdict", "REFINE")
                verdict = Verdict.from_string(verdict_raw)
                summary = synthesis.get("summary", "")

                # Gap D: remember this synthesis for future passes
                prior_syntheses.append({
                    "pass": pass_num,
                    "verdict": verdict_raw,
                    "summary": summary,
                })

                # Safety: check we actually have generated code
                has_code = self._has_generated_code(all_task_results)

                if verdict == Verdict.PASS and has_code:
                    logger.info("council.approved", pass_number=pass_num)
                    break
                elif verdict == Verdict.PASS and not has_code:
                    logger.warning("council.approve_without_code", pass_number=pass_num)
                    verdict = Verdict.PARTIAL

                # REFINE – get new tasks for next pass
                logger.info("council.refine", pass_number=pass_num)
                dag_tasks = self.orchestrator.build_refinement_tasks(
                    synthesis, complexity=complexity
                )

                # ── Gap A: inject refinement feedback into shared_ctx ──
                # The orchestrator's synthesis summary + refinement
                # instructions become the feedback for the next pass.
                refinement_instructions = []
                for rt in synthesis.get("refinement_tasks", []):
                    agent = rt.get("agent", "?")
                    instr = rt.get("instruction", "")
                    refinement_instructions.append(f"- [{agent}] {instr}")

                shared_ctx["refinement_feedback"] = (
                    f"Orchestrator verdict (pass {pass_num}): {verdict_raw}\n"
                    f"Summary: {summary}\n"
                    f"Refinement instructions:\n"
                    + "\n".join(refinement_instructions)
                )

                # ── Gap C: clear stale agent outputs from shared_ctx ──
                # Keep only durable context (language, refinement_feedback,
                # pass_number) and generated code. Remove stale reports
                # from agents that won't re-run this pass.
                next_agents = {t.agent for t in dag_tasks}
                stale_keys = [
                    k for k in list(shared_ctx)
                    if k.startswith("_report_")
                ]
                for k in stale_keys:
                    del shared_ctx[k]

                # Clear role-specific feedback if that role is re-running
                # (so it generates fresh output, not re-reads its own stale context)
                if "reviewer" in next_agents:
                    shared_ctx.pop("review_feedback", None)
                if "debugger" in next_agents:
                    shared_ctx.pop("debug_feedback", None)

                # Inject refinement context into task instructions
                for task in dag_tasks:
                    task.instruction = (
                        f"[Refinement pass {pass_num + 1}] {task.instruction}"
                    )

            else:
                # Exhausted all passes
                logger.warning("council.max_passes_reached", max=max_passes)
                if not summary:
                    summary = "Max refinement passes reached."

            # ── Build final result ──────────────────────────────────
            elapsed = time.perf_counter() - t0
            session.status = TaskStatus.COMPLETED
            session.completed_at = datetime.now(timezone.utc)

            return self.orchestrator.build_result(
                session=session,
                task_results=all_task_results,
                execution_order=all_execution_order,
                summary=summary,
                total_duration=elapsed,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            error_text = self._format_exception(exc)
            logger.error("council.failed", error=error_text, session_id=session.id)
            session.status = TaskStatus.FAILED
            return CouncilResult(
                session_id=session.id,
                status=Verdict.FAIL,
                summary=f"Council pipeline failed: {error_text}",
                total_duration_secs=round(elapsed, 2),
            )
        finally:
            await self._close_all()

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _filter_executable(self, tasks: list[DAGTask]) -> list[DAGTask]:
        """Remove tasks for agents we don't have, and fix dangling deps."""
        available = set(self.agents.keys())
        executable = [t for t in tasks if t.agent in available]
        executable_ids = {t.id for t in executable}

        # Fix deps that reference removed tasks
        for t in executable:
            t.dependencies = [d for d in t.dependencies if d in executable_ids]

        removed = [t for t in tasks if t.agent not in available]
        if removed:
            logger.warning(
                "council.tasks_filtered",
                removed=[f"{t.id}:{t.agent}" for t in removed],
                remaining=[f"{t.id}:{t.agent}" for t in executable],
            )
        return executable

    @staticmethod
    def _has_generated_code(task_results: dict[str, DAGTaskResult]) -> bool:
        """Check if any task result contains generated code with files."""
        from slm_council.models import GeneratedCode, RefactorResult

        for tr in task_results.values():
            if tr.parsed is None:
                continue
            if isinstance(tr.parsed, GeneratedCode) and tr.parsed.files:
                return True
            if isinstance(tr.parsed, RefactorResult) and tr.parsed.files:
                return True
        return False

    async def _close_all(self) -> None:
        """Close all HTTP clients."""
        coros = [self.orchestrator.close()]
        for agent in self.agents.values():
            coros.append(agent.close())
        await asyncio.gather(*coros, return_exceptions=True)

    @staticmethod
    def _format_exception(exc: Exception) -> str:
        """Return the most informative error string for nested retry exceptions."""
        if isinstance(exc, RetryError):
            last_exc = exc.last_attempt.exception()
            if last_exc is not None:
                return str(last_exc)
        return str(exc)
