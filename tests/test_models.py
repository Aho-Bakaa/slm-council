"""Tests for data models."""

from slm_council.models import (
    AgentResponse,
    AgentRole,
    AgentTask,
    BugReport,
    CodeFile,
    CouncilResult,
    CouncilSession,
    DebugReport,
    DependencySpec,
    GeneratedCode,
    RefinementFeedback,
    Severity,
    TaskStatus,
    TechManifest,
    TestCase,
    TestReport,
    UserRequest,
    Verdict,
)


def test_user_request_defaults() -> None:
    req = UserRequest(query="Build a REST API")
    assert req.query == "Build a REST API"
    assert req.language == "python"
    assert len(req.id) == 12


def test_agent_task_creation() -> None:
    task = AgentTask(
        agent=AgentRole.RESEARCHER,
        instruction="Research FastAPI best practices",
    )
    assert task.status == TaskStatus.PENDING
    assert task.agent == AgentRole.RESEARCHER


def test_tech_manifest() -> None:
    manifest = TechManifest(
        summary="Use FastAPI with Pydantic",
        dependencies=[
            DependencySpec(name="fastapi", version="0.115.0", purpose="Web framework"),
        ],
        constraints=["Python 3.11+"],
    )
    assert len(manifest.dependencies) == 1
    assert manifest.dependencies[0].name == "fastapi"


def test_generated_code() -> None:
    code = GeneratedCode(
        files=[
            CodeFile(filename="main.py", content="print('hello')", description="Entry point"),
        ],
        explanation="Simple hello world",
    )
    assert len(code.files) == 1
    assert code.files[0].language == "python"


def test_debug_report() -> None:
    report = DebugReport(
        verdict=Verdict.PARTIAL,
        bugs=[
            BugReport(
                severity=Severity.ERROR,
                description="Missing null check",
                suggested_fix="Add `if x is None` guard",
            ),
        ],
        overall_assessment="One error found",
    )
    assert report.verdict == Verdict.PARTIAL
    assert len(report.bugs) == 1


def test_test_report() -> None:
    report = TestReport(
        verdict=Verdict.PASS,
        test_cases=[
            TestCase(
                name="test_add",
                test_code="assert add(1, 2) == 3",
                expected_result="3",
            ),
        ],
        pass_count=1,
        fail_count=0,
    )
    assert report.verdict == Verdict.PASS


def test_agent_response() -> None:
    resp = AgentResponse(
        agent=AgentRole.GENERATOR,
        status=TaskStatus.COMPLETED,
        raw_output='{"files": []}',
    )
    assert resp.agent == AgentRole.GENERATOR
    assert resp.error is None


def test_council_session() -> None:
    req = UserRequest(query="Build something")
    session = CouncilSession(request=req)
    assert session.current_pass == 1
    assert session.status == TaskStatus.PENDING


def test_refinement_feedback() -> None:
    fb = RefinementFeedback(
        pass_number=1,
        source_agent=AgentRole.DEBUGGER,
        issues=["Missing error handling"],
        instructions="Add try/except blocks",
    )
    assert fb.pass_number == 1


def test_council_result() -> None:
    result = CouncilResult(
        session_id="abc123",
        status=Verdict.PASS,
        summary="All checks passed",
    )
    assert result.refinement_passes == 0
