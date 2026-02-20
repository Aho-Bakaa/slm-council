"""Council Loop – the iterative DAG-based refinement engine.

Wires together the Orchestrator brain, the Agent Registry, and the
DAG Executor into a complete dynamic pipeline with automatic refinement.

Flow:
  ┌──────────────────────────────┐
  │  User Request                │
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  Orchestrator: DECOMPOSE     │
  │  → DAG of agent tasks        │
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  DAG Executor                │
  │  • Topological wave sort     │
  │  • Parallel execution        │
  │  • Context propagation       │
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  Orchestrator: SYNTHESISE    │
  │  → APPROVE  or  REFINE ──┐  │
  └──────────┬────────────────┘  │
             │    ◄──────────────┘
             ▼
  ┌──────────────────────────────┐
  │  Return CouncilResult        │
  └──────────────────────────────┘
"""

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
    TaskStatus,
    UserRequest,
    Verdict,
)
from slm_council.orchestrator.brain import Orchestrator
from slm_council.orchestrator.dag import (
    DAGExecutionResult,
    DAGExecutor,
    DAGTask,
    DAGTaskResult,
    DAGValidationError,
)
from slm_council.utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_agents_loaded() -> None:
    """Import all agent modules so they register with the registry."""
    # Each import triggers the @registry.register decorator at module scope
    import slm_council.agents.researcher  # noqa: F401
    import slm_council.agents.planner  # noqa: F401
    import slm_council.agents.generator  # noqa: F401
    import slm_council.agents.reviewer  # noqa: F401
    import slm_council.agents.debugger  # noqa: F401
    import slm_council.agents.tester  # noqa: F401
    import slm_council.agents.optimizer  # noqa: F401
    import slm_council.agents.refactorer  # noqa: F401


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

        try:
            # Phase 1 – Orchestrator decomposes the task into a DAG
            dag_tasks = await self.orchestrator.decompose(request)
            session.status = TaskStatus.IN_PROGRESS

            # Build shared context from user request
            shared_ctx: dict[str, Any] = {
                "language": request.language,
            }

            # Create the DAG executor
            semaphore = asyncio.Semaphore(settings.max_concurrent_agent_calls)

            # ── Iteration loop ──────────────────────────────────────
            for pass_num in range(1, max_passes + 1):
                session.current_pass = pass_num
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

                # Execute the DAG
                executor = DAGExecutor(
                    agents=self.agents,
                    semaphore=semaphore,
                    session_id=session.id,
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

                # Phase 2 – Orchestrator synthesises all results
                synthesis = await self.orchestrator.synthesise(
                    session, all_task_results
                )

                verdict_raw = synthesis.get("verdict", "REFINE")
                verdict = Verdict.from_string(verdict_raw)
                summary = synthesis.get("summary", "")

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
                dag_tasks = self.orchestrator.build_refinement_tasks(synthesis)

                # Inject refinement context so agents know which pass this is
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
