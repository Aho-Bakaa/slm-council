"""DAG Execution Engine for the SLM Council.

Builds a directed acyclic graph from the orchestrator's task decomposition,
validates it (acyclicity, missing refs), then executes tasks in topological
waves — all tasks whose dependencies are met run concurrently via
``asyncio.gather``, gated by a semaphore for rate limiting.

Each completed task feeds its output into the shared context so that
downstream tasks can reference upstream results.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog

from slm_council.agents.base import BaseAgent
from slm_council.models import AgentResponse, AgentRole, ParsedAgentOutput

logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────────────


@dataclass
class DAGTask:
    """A single task node in the execution DAG."""

    id: str
    agent: str  # lowercase role name: "researcher", "generator", …
    instruction: str
    dependencies: list[str] = field(default_factory=list)


@dataclass
class DAGTaskResult:
    """Result of executing a single DAG task."""

    task_id: str
    agent: str
    response: AgentResponse | None = None
    parsed: ParsedAgentOutput = None
    error: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class DAGExecutionResult:
    """Aggregate result of running the full DAG."""

    task_results: dict[str, DAGTaskResult] = field(default_factory=dict)
    execution_order: list[list[str]] = field(default_factory=list)
    total_elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Validation helpers
# ────────────────────────────────────────────────────────────────────


class DAGValidationError(Exception):
    """Raised when the task graph is invalid."""


def validate_dag(tasks: list[DAGTask]) -> None:
    """Check for missing references, duplicates, and cycles."""
    ids = {t.id for t in tasks}

    # Duplicate IDs
    if len(ids) != len(tasks):
        seen: set[str] = set()
        dupes = []
        for t in tasks:
            if t.id in seen:
                dupes.append(t.id)
            seen.add(t.id)
        raise DAGValidationError(f"Duplicate task IDs: {dupes}")

    # Missing dependency references
    for t in tasks:
        missing = set(t.dependencies) - ids
        if missing:
            raise DAGValidationError(
                f"Task '{t.id}' references unknown dependencies: {missing}"
            )

    # Cycle detection via Kahn's algorithm
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    adj: dict[str, list[str]] = defaultdict(list)
    for t in tasks:
        for dep in t.dependencies:
            adj[dep].append(t.id)
            in_degree[t.id] += 1

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbour in adj[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if visited != len(tasks):
        raise DAGValidationError("Task graph contains a cycle")


# ────────────────────────────────────────────────────────────────────
# Topological wave computation
# ────────────────────────────────────────────────────────────────────


def topological_waves(tasks: list[DAGTask]) -> list[list[DAGTask]]:
    """Return tasks grouped into waves for parallel execution.

    Wave 0 = all tasks with no dependencies.
    Wave 1 = all tasks whose deps are only in wave 0.
    …and so on.
    """
    task_map = {t.id: t for t in tasks}
    in_degree: dict[str, int] = {t.id: len(t.dependencies) for t in tasks}
    adj: dict[str, list[str]] = defaultdict(list)
    for t in tasks:
        for dep in t.dependencies:
            adj[dep].append(t.id)

    waves: list[list[DAGTask]] = []
    ready = [tid for tid, deg in in_degree.items() if deg == 0]

    while ready:
        wave = [task_map[tid] for tid in ready]
        waves.append(wave)
        next_ready: list[str] = []
        for tid in ready:
            for neighbour in adj[tid]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    next_ready.append(neighbour)
        ready = next_ready

    return waves


# ────────────────────────────────────────────────────────────────────
# DAG Executor
# ────────────────────────────────────────────────────────────────────


class DAGExecutor:
    """Executes a validated DAG of agent tasks.

    Parameters
    ----------
    agents : dict[str, BaseAgent]
        Maps lowercase role name (e.g. "generator") to a live agent instance.
    semaphore : asyncio.Semaphore | None
        Optional concurrency limiter for API rate limiting.
    session_id : str
        Passed to agents for conversation memory.
    """

    def __init__(
        self,
        agents: dict[str, BaseAgent],
        semaphore: asyncio.Semaphore | None = None,
        session_id: str = "",
    ) -> None:
        self.agents = agents
        self.semaphore = semaphore or asyncio.Semaphore(8)
        self.session_id = session_id

    async def execute(
        self,
        tasks: list[DAGTask],
        shared_context: dict[str, Any] | None = None,
    ) -> DAGExecutionResult:
        """Run the full DAG and return aggregated results."""
        validate_dag(tasks)
        waves = topological_waves(tasks)

        ctx = dict(shared_context or {})
        result = DAGExecutionResult()
        t0 = time.perf_counter()

        for wave_idx, wave in enumerate(waves):
            wave_ids = [t.id for t in wave]
            logger.info("dag.wave_start", wave=wave_idx, tasks=wave_ids)
            result.execution_order.append(wave_ids)

            coros = [self._run_task(task, ctx) for task in wave]
            wave_results = await asyncio.gather(*coros, return_exceptions=True)

            for task, tres in zip(wave, wave_results):
                if isinstance(tres, Exception):
                    tr = DAGTaskResult(
                        task_id=task.id,
                        agent=task.agent,
                        error=str(tres),
                    )
                    result.errors.append(f"[{task.id}] {tres}")
                    logger.error("dag.task_error", task_id=task.id, error=str(tres))
                else:
                    tr = tres
                    # Feed output into context for downstream tasks
                    self._update_context(ctx, task, tr)

                result.task_results[task.id] = tr

        result.total_elapsed = time.perf_counter() - t0
        logger.info(
            "dag.complete",
            waves=len(waves),
            tasks=len(tasks),
            elapsed=f"{result.total_elapsed:.2f}s",
            errors=len(result.errors),
        )
        return result

    async def _run_task(
        self,
        task: DAGTask,
        ctx: dict[str, Any],
    ) -> DAGTaskResult:
        """Execute a single task within the semaphore gate."""
        agent = self.agents.get(task.agent)
        if agent is None:
            return DAGTaskResult(
                task_id=task.id,
                agent=task.agent,
                error=f"No agent registered for role '{task.agent}'",
            )

        async with self.semaphore:
            t0 = time.perf_counter()
            logger.info("dag.task_start", task_id=task.id, agent=task.agent)

            response = await agent.run(
                task_instruction=task.instruction,
                session_id=self.session_id,
                **ctx,
            )
            elapsed = time.perf_counter() - t0

            logger.info(
                "dag.task_done",
                task_id=task.id,
                agent=task.agent,
                elapsed=f"{elapsed:.2f}s",
            )

            return DAGTaskResult(
                task_id=task.id,
                agent=task.agent,
                response=response,
                parsed=response.parsed_output,
                elapsed_seconds=elapsed,
            )

    @staticmethod
    def _update_context(
        ctx: dict[str, Any],
        task: DAGTask,
        result: DAGTaskResult,
    ) -> None:
        """Inject task output into shared context for downstream tasks.

        Uses role-based keys so that downstream agents can naturally pick
        up upstream outputs via their standard ``ctx.get(...)`` calls.
        """
        if result.response is None:
            return

        role = task.agent.lower()
        parsed = result.parsed

        if role == "researcher" and parsed is not None:
            # Downstream agents expect 'tech_manifest' as a JSON string
            try:
                ctx["tech_manifest"] = parsed.model_dump_json(indent=2)
            except Exception:
                ctx["tech_manifest"] = json.dumps({"summary": str(parsed)})

        elif role == "planner" and parsed is not None:
            try:
                ctx["architecture_plan"] = parsed.model_dump_json(indent=2)
            except Exception:
                ctx["architecture_plan"] = json.dumps({"summary": str(parsed)})

        elif role == "generator" and parsed is not None:
            # Code agents expect 'code' as a string (all file contents joined)
            try:
                files = getattr(parsed, "files", [])
                code_str = "\n\n".join(
                    f"# {f.filename}\n{f.content}" for f in files if f.content
                )
                ctx["code"] = code_str or str(parsed)
            except Exception:
                ctx["code"] = str(parsed)

        elif role == "reviewer" and parsed is not None:
            ctx["review_feedback"] = str(parsed)

        elif role == "debugger" and parsed is not None:
            ctx["debug_feedback"] = str(parsed)

        elif role == "refactorer" and parsed is not None:
            # Refactored code replaces the current code context
            try:
                files = getattr(parsed, "files", [])
                if files:
                    code_str = "\n\n".join(
                        f"# {f.filename}\n{f.content}" for f in files if f.content
                    )
                    ctx["code"] = code_str
            except Exception:
                pass

        # Always store under role-specific key for brain.py to collect
        ctx[f"_report_{role}_{task.id}"] = result
