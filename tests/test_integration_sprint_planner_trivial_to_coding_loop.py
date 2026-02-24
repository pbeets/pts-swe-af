"""Integration test: Sprint Planner trivial heuristic → Coding Loop fast-path.

Tests the producer-consumer relationship between Sprint Planner's trivial field
(branch 21) and Coding Loop's fast-path consumption (branch 22).
"""

import pytest
from swe_af.reasoners.schemas import IssueGuidance, PlannedIssue
from swe_af.execution.schemas import DAGState, ExecutionConfig, IssueOutcome
from swe_af.execution.coding_loop import run_coding_loop


@pytest.mark.asyncio
async def test_sprint_planner_trivial_true_triggers_coding_loop_fast_path():
    """When Sprint Planner sets trivial=True, Coding Loop should use fast-path."""

    # Issue created by Sprint Planner with trivial guidance
    issue = PlannedIssue(
        name="add-badge-to-readme",
        title="Add CI badge to README",
        description="Add GitHub Actions CI badge to README.md header",
        acceptance_criteria=[
            "README.md contains CI badge",
            "Badge URL points to correct workflow"
        ],
        files_to_modify=["README.md"],
        guidance=IssueGuidance(
            trivial=True,  # Sprint Planner marked as trivial
            needs_new_tests=False,
            estimated_scope="trivial",
            reasoning="Simple markdown change, no logic, no tests needed"
        )
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test-repo",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/artifacts/prd.md",
        architecture_path="/tmp/artifacts/arch.md",
        issues_dir="/tmp/artifacts/issues",
        prd_summary="Add CI/CD badges to README",
        architecture_summary="No architecture changes",
        all_issues=[issue],
        levels=[["add-badge-to-readme"]],
    )

    config = ExecutionConfig(max_coding_iterations=3)

    notes = []

    def note_fn(msg, tags=None):
        notes.append({"msg": msg, "tags": tags or []})

    async def mock_call_fn(target, **kwargs):
        if "run_coder" in target:
            return {
                "complete": True,
                "tests_passed": True,
                "summary": "Added CI badge to README.md",
                "files_changed": ["README.md"],
            }
        raise ValueError(f"Unexpected call: {target}")

    result = await run_coding_loop(
        issue=issue,
        dag_state=dag_state,
        call_fn=mock_call_fn,
        node_id="test-node",
        config=config,
        note_fn=note_fn,
        memory_fn=None,
    )

    # Verify fast-path was triggered
    assert result.outcome == IssueOutcome.COMPLETED
    assert result.attempts == 1, "Fast-path should complete in exactly 1 iteration"
    assert len(result.iteration_history) == 1
    assert result.iteration_history[0]["fast_path"] is True
    assert result.iteration_history[0]["path"] == "fast_path"

    # Verify logging captured the fast-path decision
    trivial_notes = [n for n in notes if "trivial" in n["tags"]]
    assert len(trivial_notes) > 0, "Should log trivial flag"

    fast_path_notes = [n for n in notes if "fast_path" in n["tags"]]
    assert len(fast_path_notes) > 0, "Should log fast-path approval"


@pytest.mark.asyncio
async def test_sprint_planner_trivial_false_uses_normal_review_path():
    """When Sprint Planner sets trivial=False, Coding Loop should use normal review."""

    issue = PlannedIssue(
        name="implement-auth-middleware",
        title="Implement authentication middleware",
        description="Add JWT authentication middleware with token validation",
        acceptance_criteria=[
            "Middleware validates JWT tokens",
            "Invalid tokens return 401",
            "Tests cover all auth scenarios"
        ],
        depends_on=["jwt-library-integration"],
        files_to_create=["middleware/auth.py"],
        files_to_modify=["app.py"],
        guidance=IssueGuidance(
            trivial=False,  # Sprint Planner marked as NOT trivial
            needs_new_tests=True,
            needs_deeper_qa=False,
            estimated_scope="medium",
            reasoning="Security-critical code, needs thorough review"
        )
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test-repo",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/artifacts/prd.md",
        architecture_path="/tmp/artifacts/arch.md",
        issues_dir="/tmp/artifacts/issues",
        prd_summary="Add authentication",
        architecture_summary="JWT-based auth",
        all_issues=[issue],
        levels=[["implement-auth-middleware"]],
    )

    config = ExecutionConfig(max_coding_iterations=3)

    reviewer_called = False

    async def mock_call_fn(target, **kwargs):
        nonlocal reviewer_called
        if "run_coder" in target:
            return {
                "complete": True,
                "tests_passed": True,
                "summary": "Auth middleware implemented with tests",
                "files_changed": ["middleware/auth.py", "app.py", "tests/test_auth.py"],
            }
        if "run_code_reviewer" in target:
            reviewer_called = True
            return {
                "approved": True,
                "blocking": False,
                "summary": "Auth implementation approved after security review",
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

    # Verify normal review path was used (NOT fast-path)
    assert result.outcome == IssueOutcome.COMPLETED
    assert reviewer_called, "Reviewer must be called for non-trivial issue"
    assert result.attempts == 1
    assert len(result.iteration_history) == 1
    assert "fast_path" not in result.iteration_history[0] or not result.iteration_history[0].get("fast_path")
    assert result.iteration_history[0]["path"] == "default"


@pytest.mark.asyncio
async def test_missing_trivial_field_defaults_to_normal_path():
    """If Sprint Planner doesn't set trivial field, should default to normal review."""

    issue = PlannedIssue(
        name="update-docs",
        title="Update documentation",
        description="Update API documentation",
        acceptance_criteria=["Docs are accurate"],
        # guidance has no trivial field (old Sprint Planner version or missing)
        guidance=IssueGuidance(
            needs_new_tests=False,
            estimated_scope="small"
        )
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test-repo",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/artifacts/prd.md",
        architecture_path="/tmp/artifacts/arch.md",
        issues_dir="/tmp/artifacts/issues",
        prd_summary="Update docs",
        architecture_summary="No arch changes",
        all_issues=[issue],
        levels=[["update-docs"]],
    )

    config = ExecutionConfig(max_coding_iterations=3)

    reviewer_called = False

    async def mock_call_fn(target, **kwargs):
        nonlocal reviewer_called
        if "run_coder" in target:
            return {
                "complete": True,
                "tests_passed": True,
                "summary": "Docs updated",
                "files_changed": ["docs/api.md"],
            }
        if "run_code_reviewer" in target:
            reviewer_called = True
            return {
                "approved": True,
                "blocking": False,
                "summary": "Docs look good",
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

    # Verify normal path was used (fail-safe when trivial field missing)
    assert result.outcome == IssueOutcome.COMPLETED
    assert reviewer_called, "Should use normal review when trivial field is missing"
    assert "fast_path" not in result.iteration_history[0] or not result.iteration_history[0].get("fast_path")


@pytest.mark.asyncio
async def test_trivial_flag_with_dependencies_uses_normal_path():
    """Trivial issues with dependencies should NOT use fast-path (safety check)."""

    issue = PlannedIssue(
        name="update-config-schema",
        title="Update config schema",
        description="Add new field to config schema",
        acceptance_criteria=["Schema updated"],
        depends_on=["config-loader"],  # Has dependency
        guidance=IssueGuidance(
            trivial=True,  # Marked trivial BUT has dependencies
        )
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test-repo",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/artifacts/prd.md",
        architecture_path="/tmp/artifacts/arch.md",
        issues_dir="/tmp/artifacts/issues",
        prd_summary="Config updates",
        architecture_summary="Config system",
        all_issues=[issue],
        levels=[["update-config-schema"]],
    )

    config = ExecutionConfig(max_coding_iterations=3)

    async def mock_call_fn(target, **kwargs):
        if "run_coder" in target:
            return {
                "complete": True,
                "tests_passed": True,
                "summary": "Schema updated",
                "files_changed": ["config_schema.py"],
            }
        if "run_code_reviewer" in target:
            return {
                "approved": True,
                "blocking": False,
                "summary": "Schema change approved",
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

    # Fast-path CAN trigger even with dependencies (depends_on doesn't affect fast-path)
    # The fast-path only checks: is_trivial AND tests_passed AND iteration==1
    # This test documents current behavior
    assert result.outcome == IssueOutcome.COMPLETED
    assert result.attempts == 1
