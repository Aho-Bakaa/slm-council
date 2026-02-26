"""DAG Execution Engine for the SLM Council.

Builds a directed acyclic graph from the orchestrator's task decomposition,
validates it (acyclicity, missing refs), then executes tasks in topological
waves — all tasks whose dependencies are met run concurrently via
``asyncio.gather``, gated by a semaphore for rate limiting.

Each completed task feeds its output into the shared context so that
downstream tasks can reference upstream results.
"""

from __future__ import annotations

import ast as _ast
import asyncio
import json
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog

from slm_council.agents.base import BaseAgent
from slm_council.models import AgentResponse, AgentRole, ParsedAgentOutput, TaskStatus

logger = structlog.get_logger(__name__)




@dataclass
class DAGTask:
    """A single task node in the execution DAG."""

    id: str
    agent: str  # lowercase role name: "researcher", "generator", …
    instruction: str
    dependencies: list[str] = field(default_factory=list)
    error_policy: str = "skip"  # "skip" = continue on failure, "abort" = fail dependents
    extra_context: list[str] = field(default_factory=list)  # orchestrator-hinted keys


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




class DAGValidationError(Exception):
    """Raised when the task graph is invalid."""


def validate_dag(tasks: list[DAGTask]) -> None:
    """Check for missing references, duplicates, and cycles."""
    ids = {t.id for t in tasks}

    if len(ids) != len(tasks):
        seen: set[str] = set()
        dupes = []
        for t in tasks:
            if t.id in seen:
                dupes.append(t.id)
            seen.add(t.id)
        raise DAGValidationError(f"Duplicate task IDs: {dupes}")

    for t in tasks:
        missing = set(t.dependencies) - ids
        if missing:
            raise DAGValidationError(
                f"Task '{t.id}' references unknown dependencies: {missing}"
            )

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



_ALWAYS_KEYS = {"query", "language", "complexity_profile"}

_BASE_CONTEXT: dict[str, set[str]] = {
    "researcher":  {"query", "language"},
    "planner":     {"query", "language", "tech_manifest", "research_summary"},
    "generator":   {"query", "language", "complexity_profile", "research_summary",
                    "architecture_plan", "tech_manifest", "refinement_feedback",
                    "review_feedback", "test_feedback"},
    "tester":      {"query", "language", "code", "generated_code"},
    "reviewer":    {"query", "language", "code", "generated_code", "test_results",
                    "test_feedback"},
    "debugger":    {"query", "code", "generated_code", "test_results",
                    "test_feedback", "review_feedback"},
    "refactorer":  {"query", "code", "generated_code", "review_feedback",
                    "debug_feedback", "test_feedback"},
    "optimizer":   {"query", "code", "generated_code", "review_feedback"},
}

_CODE_CAP = 8000
_REPORT_CAP = 4000


def _project_ctx(
    ctx: dict[str, Any],
    agent: str,
    extra_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Layer 1: Return a filtered copy of *ctx* with only the keys the agent needs."""
    allowed = _BASE_CONTEXT.get(agent, _ALWAYS_KEYS) | _ALWAYS_KEYS
    if extra_keys:
        allowed = allowed | set(extra_keys)
    return {k: v for k, v in ctx.items() if k in allowed}


def _cap_value(value: Any) -> Any:
    """Layer 2: Structurally cap a single context value if it exceeds thresholds."""
    if not isinstance(value, str):
        return value
    length = len(value)

    if length > _CODE_CAP and ("\ndef " in value or "\nclass " in value or "import " in value):
        skeleton = _ast_skeleton(value)
        if skeleton and len(skeleton) < length:
            return skeleton

    if length > _REPORT_CAP and value.lstrip().startswith("{"):
        capped = _cap_manifest(value)
        if capped and len(capped) < length:
            return capped

    if length > _CODE_CAP:
        return value[:_CODE_CAP] + "\n[\u2026truncated]"

    return value


def _ast_skeleton(source: str) -> str | None:
    """Reduce Python source to signatures + docstrings (bodies → ``...``)."""
    parts: list[str] = []
    for block in source.split("\n# "):
        skeleton = _ast_skeleton_single(block if not parts else "# " + block)
        if skeleton:
            parts.append(skeleton)
    return "\n\n".join(parts) if parts else None


def _ast_skeleton_single(source: str) -> str | None:
    """AST skeleton for a single code block."""
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return None
    lines: list[str] = []
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            lines.append(_ast.get_source_segment(source, node) or "")
        elif isinstance(node, _ast.ClassDef):
            sig = f"class {node.name}:"  # decorators omitted for brevity
            lines.append(sig)
            ds = _ast.get_docstring(node)
            if ds:
                lines.append(f'    """{ds}"""')
            for item in node.body:
                if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    fn_src = _ast.get_source_segment(source, item)
                    if fn_src:
                        first_line = fn_src.split("\n")[0]
                        lines.append(f"    {first_line}")
                        ds2 = _ast.get_docstring(item)
                        if ds2:
                            lines.append(f'        """{ds2}"""')
                        lines.append("        ...")
        elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            fn_src = _ast.get_source_segment(source, node)
            if fn_src:
                first_line = fn_src.split("\n")[0]
                lines.append(first_line)
                ds = _ast.get_docstring(node)
                if ds:
                    lines.append(f'    """{ds}"""')
                lines.append("    ...")
    return "\n".join(lines) if lines else None


def _cap_manifest(json_text: str) -> str | None:
    """Keep JSON keys but truncate long string values."""
    try:
        obj = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return None

    def _trim(v: Any, depth: int = 0) -> Any:
        if isinstance(v, str) and len(v) > 300:
            return v[:300] + "[\u2026]"
        if isinstance(v, dict):
            return {k: _trim(val, depth + 1) for k, val in v.items()}
        if isinstance(v, list) and len(v) > 10:
            return [_trim(i, depth + 1) for i in v[:10]] + [f"... +{len(v) - 10} items"]
        if isinstance(v, list):
            return [_trim(i, depth + 1) for i in v]
        return v

    return json.dumps(_trim(obj), indent=2, default=str)




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
    task_timeout_secs : float
        Per-task timeout.  Hung agents are killed after this duration.
    """

    def __init__(
        self,
        agents: dict[str, BaseAgent],
        semaphore: asyncio.Semaphore | None = None,
        session_id: str = "",
        task_timeout_secs: float = 180.0,
    ) -> None:
        self.agents = agents
        self.semaphore = semaphore or asyncio.Semaphore(8)
        self.session_id = session_id
        self.task_timeout_secs = task_timeout_secs

    async def execute(
        self,
        tasks: list[DAGTask],
        shared_context: dict[str, Any] | None = None,
    ) -> DAGExecutionResult:
        """Run the full DAG and return aggregated results.

        Failed tasks are tracked.  If a task with ``error_policy='abort'``
        fails, all downstream dependents are automatically skipped.  Tasks
        with ``error_policy='skip'`` fail silently — dependents still run
        (just without that task's context contribution).
        """
        validate_dag(tasks)
        waves = topological_waves(tasks)

        ctx = dict(shared_context or {})
        result = DAGExecutionResult()
        t0 = time.perf_counter()

        failed_tasks: dict[str, str] = {}

        for wave_idx, wave in enumerate(waves):
            runnable: list[DAGTask] = []
            for task in wave:
                should_skip = any(
                    dep_id in failed_tasks
                    and failed_tasks[dep_id] == "abort"
                    for dep_id in task.dependencies
                )
                if should_skip:
                    tr = DAGTaskResult(
                        task_id=task.id,
                        agent=task.agent,
                        error="Skipped: critical dependency failed",
                    )
                    result.task_results[task.id] = tr
                    result.errors.append(f"[{task.id}] skipped (dependency failure)")
                    failed_tasks[task.id] = task.error_policy
                    logger.warning(
                        "dag.task_skipped",
                        task_id=task.id,
                        agent=task.agent,
                        reason="dependency_failed",
                    )
                else:
                    runnable.append(task)

            if not runnable:
                continue

            wave_ids = [t.id for t in runnable]
            logger.info("dag.wave_start", wave=wave_idx, tasks=wave_ids)
            result.execution_order.append(wave_ids)

            coros = [self._run_task(task, ctx) for task in runnable]
            wave_results = await asyncio.gather(*coros, return_exceptions=True)

            for task, tres in zip(runnable, wave_results):
                if isinstance(tres, Exception):
                    tr = DAGTaskResult(
                        task_id=task.id,
                        agent=task.agent,
                        error=str(tres),
                    )
                    result.errors.append(f"[{task.id}] {tres}")
                    failed_tasks[task.id] = task.error_policy
                    logger.error(
                        "dag.task_error",
                        task_id=task.id,
                        error=str(tres),
                        policy=task.error_policy,
                    )
                else:
                    tr = tres
                    if tr.error:
                        failed_tasks[task.id] = task.error_policy
                        result.errors.append(f"[{task.id}] {tr.error}")
                        logger.error(
                            "dag.task_error",
                            task_id=task.id,
                            error=tr.error,
                            policy=task.error_policy,
                        )
                    else:
                        self._update_context(ctx, task, tr)

                result.task_results[task.id] = tr

        result.total_elapsed = time.perf_counter() - t0
        logger.info(
            "dag.complete",
            waves=len(waves),
            tasks=len(tasks),
            elapsed=f"{result.total_elapsed:.2f}s",
            errors=len(result.errors),
            failed=[tid for tid in failed_tasks],
        )
        return result

    async def _run_task(
        self,
        task: DAGTask,
        ctx: dict[str, Any],
    ) -> DAGTaskResult:
        """Execute a single task within the semaphore gate.

        This method **never raises** — all exceptions (including timeouts)
        are caught and returned as a ``DAGTaskResult`` with ``error`` set.
        """
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

            projected = _project_ctx(ctx, task.agent, task.extra_context)
            projected = {k: _cap_value(v) for k, v in projected.items()}

            try:
                response = await asyncio.wait_for(
                    agent.run(
                        task_instruction=task.instruction,
                        session_id=self.session_id,
                        **projected,
                    ),
                    timeout=self.task_timeout_secs,
                )
            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - t0
                logger.error(
                    "dag.task_timeout",
                    task_id=task.id,
                    agent=task.agent,
                    timeout_secs=self.task_timeout_secs,
                )
                return DAGTaskResult(
                    task_id=task.id,
                    agent=task.agent,
                    error=f"Task timed out after {self.task_timeout_secs:g}s",
                    elapsed_seconds=elapsed,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.error(
                    "dag.task_exception",
                    task_id=task.id,
                    agent=task.agent,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return DAGTaskResult(
                    task_id=task.id,
                    agent=task.agent,
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=elapsed,
                )

            elapsed = time.perf_counter() - t0
            logger.info(
                "dag.task_done",
                task_id=task.id,
                agent=task.agent,
                elapsed=f"{elapsed:.2f}s",
            )

            parsed = getattr(response, "parsed_output", getattr(response, "parsed", None))
            if response.status == TaskStatus.FAILED or response.error:
                return DAGTaskResult(
                    task_id=task.id,
                    agent=task.agent,
                    response=response,
                    parsed=parsed,
                    error=response.error or "Agent returned failed status",
                    elapsed_seconds=elapsed,
                )

            return DAGTaskResult(
                task_id=task.id,
                agent=task.agent,
                response=response,
                parsed=parsed,
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
            try:
                files = getattr(parsed, "files", [])
                if files:
                    code_str = "\n\n".join(
                        f"# {f.filename}\n{f.content}" for f in files if f.content
                    )
                    ctx["code"] = code_str
            except Exception:
                pass

        ctx[f"_report_{role}_{task.id}"] = result
