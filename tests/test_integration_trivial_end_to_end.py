"""Integration test: Sprint Planner trivial flag → Coding Loop fast-path.

Tests the end-to-end flow where Sprint Planner sets guidance.trivial=True,
and the coding loop correctly consumes it to trigger fast-path approval.
"""

import pytest
from swe_af.reasoners.schemas import IssueGuidance, PlannedIssue
from swe_af.execution.schemas import DAGState, ExecutionConfig, IssueOutcome
from swe_af.execution.coding_loop import run_coding_loop


@pytest.mark.asyncio
async def test_trivial_flag_integration_end_to_end():
    """Sprint Planner trivial=True → Coding Loop fast-path approval."""

    # Issue with trivial guidance from Sprint Planner
    issue = PlannedIssue(
        name="update-readme",
        title="Update README with new instructions",
        description="Add installation instructions to README.md",
        acceptance_criteria=["README.md exists with installation section"],
        guidance=IssueGuidance(
            trivial=True,  # Sprint Planner marked this trivial
            needs_new_tests=False,
            estimated_scope="trivial"
        )
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test-repo",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/artifacts/prd.md",
        architecture_path="/tmp/artifacts/arch.md",
        issues_dir="/tmp/artifacts/issues",
        prd_summary="Test PRD",
        architecture_summary="Test Architecture",
        all_issues=[issue],
        levels=[["update-readme"]],
    )

    config = ExecutionConfig(max_coding_iterations=3)

    # Mock call_fn that simulates coder returning passing tests
    async def mock_call_fn(target, **kwargs):
        if "run_coder" in target:
            return {
                "complete": True,
                "tests_passed": True,
                "summary": "Updated README.md with installation instructions",
                "files_changed": ["README.md"],
            }
        raise ValueError(f"Unexpected call: {target}")

    # Run coding loop
    result = await run_coding_loop(
        issue=issue,
        dag_state=dag_state,
        call_fn=mock_call_fn,
        node_id="test-node",
        config=config,
        note_fn=None,
        memory_fn=None,
    )

    # Verify fast-path was triggered
    assert result.outcome == IssueOutcome.COMPLETED
    assert result.attempts == 1, "Fast-path should complete in 1 iteration"
    assert len(result.iteration_history) == 1
    assert result.iteration_history[0]["fast_path"] is True
    assert result.iteration_history[0]["action"] == "approve"


@pytest.mark.asyncio
async def test_trivial_flag_with_failing_tests_skips_fast_path():
    """Trivial issue with failing tests should NOT trigger fast-path."""

    issue = PlannedIssue(
        name="update-config",
        title="Update config file",
        description="Add new config option",
        acceptance_criteria=["Config file updated"],
        guidance=IssueGuidance(trivial=True)
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test-repo",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/artifacts/prd.md",
        architecture_path="/tmp/artifacts/arch.md",
        issues_dir="/tmp/artifacts/issues",
        prd_summary="Test PRD",
        architecture_summary="Test Architecture",
        all_issues=[issue],
        levels=[["update-config"]],
    )

    config = ExecutionConfig(max_coding_iterations=3)

    call_count = 0

    async def mock_call_fn(target, **kwargs):
        nonlocal call_count
        if "run_coder" in target:
            call_count += 1
            if call_count == 1:
                # First iteration: tests fail
                return {
                    "complete": True,
                    "tests_passed": False,
                    "summary": "Config updated but tests failed",
                    "files_changed": ["config.yml"],
                }
            else:
                # Second iteration: tests pass
                return {
                    "complete": True,
                    "tests_passed": True,
                    "summary": "Config updated, tests passing",
                    "files_changed": ["config.yml"],
                }
        if "run_code_reviewer" in target:
            return {
                "approved": True,
                "blocking": False,
                "summary": "LGTM",
            }
        raise ValueError(f"Unexpected call: {target}")

    result = await run_coding_loop(
        issue=issue,
        dag_state=dag_state,
        call_fn=mock_call_fn,
        node_id="test-node",
        config=config,
        note_fn=None,
        memory_fn=None,
    )

    # Fast-path should NOT trigger (tests failed on iteration 1)
    assert result.outcome == IssueOutcome.COMPLETED
    assert result.attempts == 2, "Should need 2 iterations (no fast-path)"
    # First iteration should NOT have fast_path flag
    assert "fast_path" not in result.iteration_history[0] or not result.iteration_history[0].get("fast_path")


@pytest.mark.asyncio
async def test_non_trivial_issue_never_triggers_fast_path():
    """Non-trivial issue should never trigger fast-path, even with passing tests."""

    issue = PlannedIssue(
        name="implement-parser",
        title="Implement parser module",
        description="Complex parser implementation",
        acceptance_criteria=["Parser handles all syntax", "Tests pass", "Performance benchmarks met"],
        depends_on=["lexer"],
        guidance=IssueGuidance(
            trivial=False,  # NOT trivial
            needs_deeper_qa=False
        )
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test-repo",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/artifacts/prd.md",
        architecture_path="/tmp/artifacts/arch.md",
        issues_dir="/tmp/artifacts/issues",
        prd_summary="Test PRD",
        architecture_summary="Test Architecture",
        all_issues=[issue],
        levels=[["implement-parser"]],
    )

    config = ExecutionConfig(max_coding_iterations=3)

    async def mock_call_fn(target, **kwargs):
        if "run_coder" in target:
            return {
                "complete": True,
                "tests_passed": True,
                "summary": "Parser implemented",
                "files_changed": ["parser.py", "tests/test_parser.py"],
            }
        if "run_code_reviewer" in target:
            return {
                "approved": True,
                "blocking": False,
                "summary": "Implementation looks good",
            }
        raise ValueError(f"Unexpected call: {target}")

    result = await run_coding_loop(
        issue=issue,
        dag_state=dag_state,
        call_fn=mock_call_fn,
        node_id="test-node",
        config=config,
        note_fn=None,
        memory_fn=None,
    )

    # Should complete normally through reviewer path
    assert result.outcome == IssueOutcome.COMPLETED
    assert result.attempts == 1
    # Should NOT have fast_path flag
    assert len(result.iteration_history) == 1
    assert "fast_path" not in result.iteration_history[0] or not result.iteration_history[0].get("fast_path")
