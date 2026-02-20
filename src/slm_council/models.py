"""Pydantic domain models shared across the entire council pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────
# Enums
# ────────────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    """Canonical names for every agent in the council."""

    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    PLANNER = "planner"
    GENERATOR = "generator"
    REVIEWER = "reviewer"
    DEBUGGER = "debugger"
    TESTER = "tester"
    OPTIMIZER = "optimizer"
    REFACTORER = "refactorer"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REFINEMENT = "needs_refinement"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"

    @classmethod
    def from_string(cls, value: str) -> "Verdict":
        """Normalize string to Verdict enum, defaulting to PARTIAL."""
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            if normalized in ("approve", "approved", "ok", "success"):
                return cls.PASS
            if normalized in ("reject", "rejected", "error"):
                return cls.FAIL
            return cls.PARTIAL


class ParseErrorCategory(str, Enum):
    """Categories of parse failures for structured error reporting."""

    JSON_MALFORMED = "json_malformed"
    SCHEMA_MISMATCH = "schema_mismatch"
    EMPTY_RESPONSE = "empty_response"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    API_ERROR = "api_error"
    UNKNOWN = "unknown"


# ────────────────────────────────────────────────────────────────────
# User Request
# ────────────────────────────────────────────────────────────────────

class UserRequest(BaseModel):
    """Incoming request from the user."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    query: str
    language: str = "python"
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ────────────────────────────────────────────────────────────────────
# Orchestrator → Agent task assignment
# ────────────────────────────────────────────────────────────────────

class AgentTask(BaseModel):
    """A discrete unit of work the Orchestrator assigns to an agent."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent: AgentRole
    instruction: str
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    context: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────
# Tech Researcher output
# ────────────────────────────────────────────────────────────────────

class DependencySpec(BaseModel):
    name: str
    version: str = ""
    purpose: str = ""


class TechManifest(BaseModel):
    """Structured research output produced by the Tech Researcher."""

    summary: str
    architecture_notes: str = ""
    dependencies: list[DependencySpec] = Field(default_factory=list)
    api_contracts: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    design_patterns: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Code Generator output
# ────────────────────────────────────────────────────────────────────

class CodeFile(BaseModel):
    """A single generated source file."""

    filename: str
    language: str = "python"
    content: str
    description: str = ""


class GeneratedCode(BaseModel):
    """Complete output from the Code Generator."""

    files: list[CodeFile]
    explanation: str = ""
    assumptions: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Debugger output
# ────────────────────────────────────────────────────────────────────

class BugReport(BaseModel):
    """A single issue found by the Debugger."""

    file: str = ""
    line_range: str = ""
    severity: Severity = Severity.WARNING
    category: str = ""
    description: str
    suggested_fix: str = ""


class DebugReport(BaseModel):
    """Full analysis from the Debugger agent."""

    verdict: Verdict
    reasoning_trace: str = ""
    bugs: list[BugReport] = Field(default_factory=list)
    overall_assessment: str = ""


# ────────────────────────────────────────────────────────────────────
# Tester output
# ────────────────────────────────────────────────────────────────────

class TestCase(BaseModel):
    """A single generated test case."""

    name: str
    description: str = ""
    test_code: str
    expected_result: str = ""


class TestReport(BaseModel):
    """Full QA report from the Tester agent."""

    verdict: Verdict
    test_cases: list[TestCase] = Field(default_factory=list)
    edge_cases_identified: list[str] = Field(default_factory=list)
    coverage_notes: str = ""
    pass_count: int = 0
    fail_count: int = 0
    failure_details: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Planner / Architect output
# ────────────────────────────────────────────────────────────────────

class ComponentSpec(BaseModel):
    """A single component in the planned architecture."""

    name: str
    responsibility: str = ""
    interfaces: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class ArchitecturePlan(BaseModel):
    """Structured design output from the Planner agent."""

    summary: str
    components: list[ComponentSpec] = Field(default_factory=list)
    file_layout: list[str] = Field(default_factory=list)
    class_hierarchy: str = ""
    api_design: str = ""
    data_flow: str = ""
    design_decisions: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Reviewer output
# ────────────────────────────────────────────────────────────────────

class StyleIssue(BaseModel):
    """A code style or best-practice violation."""

    file: str = ""
    line_range: str = ""
    category: str = ""  # naming, formatting, anti-pattern, security, etc.
    description: str
    suggestion: str = ""


class ReviewReport(BaseModel):
    """Code review report from the Reviewer agent."""

    verdict: Verdict
    style_issues: list[StyleIssue] = Field(default_factory=list)
    security_issues: list[StyleIssue] = Field(default_factory=list)
    best_practice_violations: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    overall_assessment: str = ""


# ────────────────────────────────────────────────────────────────────
# Optimizer output
# ────────────────────────────────────────────────────────────────────

class Bottleneck(BaseModel):
    """A performance bottleneck identified by the Optimizer."""

    location: str = ""  # file:line or function name
    description: str
    current_complexity: str = ""  # e.g. O(n^2)
    suggested_complexity: str = ""  # e.g. O(n log n)
    improvement: str = ""


class OptimizationReport(BaseModel):
    """Performance analysis from the Optimizer agent."""

    verdict: Verdict
    complexity_analysis: str = ""
    bottlenecks: list[Bottleneck] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    memory_notes: str = ""
    overall_assessment: str = ""


# ────────────────────────────────────────────────────────────────────
# Refactorer output
# ────────────────────────────────────────────────────────────────────

class RefactorResult(BaseModel):
    """Restructured code from the Refactorer agent."""

    files: list[CodeFile] = Field(default_factory=list)
    changes_made: list[str] = Field(default_factory=list)
    patterns_applied: list[str] = Field(default_factory=list)
    explanation: str = ""


# ────────────────────────────────────────────────────────────────────
# Agent generic response wrapper
# ────────────────────────────────────────────────────────────────────

# Union of all possible parsed agent outputs
ParsedAgentOutput = (
    TechManifest
    | ArchitecturePlan
    | GeneratedCode
    | ReviewReport
    | DebugReport
    | TestReport
    | OptimizationReport
    | RefactorResult
    | None
)


class AgentResponse(BaseModel):
    """Uniform wrapper returned by every agent."""

    agent: AgentRole
    task_id: str = ""
    status: TaskStatus = TaskStatus.COMPLETED
    raw_output: str = ""
    parsed: ParsedAgentOutput = None
    error: str | None = None
    error_category: ParseErrorCategory | None = None
    duration_secs: float = 0.0
    token_usage: dict[str, Any] = Field(default_factory=dict)
    cache_hit: bool = False
    request_hash: str = ""


# ────────────────────────────────────────────────────────────────────
# Council session (full pipeline state)
# ────────────────────────────────────────────────────────────────────

class RefinementFeedback(BaseModel):
    """Feedback the Orchestrator attaches when sending work back."""

    pass_number: int
    source_agent: AgentRole  # The agent whose output was reviewed
    target_agent: AgentRole | None = None  # The agent that should act on feedback
    issues: list[str]
    instructions: str
    # Structured references for preserving context across passes
    bug_refs: list[str] = Field(default_factory=list)  # Bug IDs from debugger
    test_refs: list[str] = Field(default_factory=list)  # Failed test names
    severity: Severity = Severity.WARNING


class CouncilSession(BaseModel):
    """Top-level state object tracking one end-to-end pipeline run."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    request: UserRequest
    tasks: list[AgentTask] = Field(default_factory=list)
    responses: list[AgentResponse] = Field(default_factory=list)
    refinements: list[RefinementFeedback] = Field(default_factory=list)
    current_pass: int = 1
    status: TaskStatus = TaskStatus.PENDING
    final_output: GeneratedCode | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


# ────────────────────────────────────────────────────────────────────
# Final response to the user
# ────────────────────────────────────────────────────────────────────

class CouncilResult(BaseModel):
    """The polished response returned to the caller."""

    session_id: str
    status: Verdict
    code: GeneratedCode | None = None
    tech_manifest: TechManifest | None = None
    architecture_plan: ArchitecturePlan | None = None
    debug_report: DebugReport | None = None
    test_report: TestReport | None = None
    review_report: ReviewReport | None = None
    optimization_report: OptimizationReport | None = None
    refactored_code: RefactorResult | None = None
    refinement_passes: int = 0
    total_duration_secs: float = 0.0
    summary: str = ""
    # Execution metadata
    agents_used: list[AgentRole] = Field(default_factory=list)
    dag_execution_order: list[list[str]] = Field(default_factory=list)  # Waves of task IDs
