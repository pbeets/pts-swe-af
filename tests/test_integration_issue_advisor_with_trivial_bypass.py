"""Integration test: Issue Advisor interaction with trivial fast-path.

Tests that when trivial fast-path is taken, Issue Advisor is correctly bypassed,
and when normal path is taken, Issue Advisor threshold logic (branch 19) works.
"""

import pytest
from swe_af.reasoners.schemas import IssueGuidance, PlannedIssue
from swe_af.execution.schemas import DAGState, ExecutionConfig, IssueOutcome
from swe_af.execution.dag_executor import _execute_single_issue


@pytest.mark.asyncio
async def test_trivial_fast_path_never_invokes_issue_advisor():
    """Trivial fast-path should complete without ever invoking Issue Advisor."""

    issue = PlannedIssue(
        name="trivial-fix",
        title="Fix typo in docs",
        description="Correct spelling error",
        acceptance_criteria=["Typo fixed"],
        guidance=IssueGuidance(trivial=True)
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test-repo",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/artifacts/prd.md",
        architecture_path="/tmp/artifacts/arch.md",
        issues_dir="/tmp/artifacts/issues",
        prd_summary="Fix docs",
        architecture_summary="No changes",
        all_issues=[issue],
        levels=[["trivial-fix"]],
    )

    config = ExecutionConfig(
        max_coding_iterations=6,
        enable_issue_advisor=True,
        max_advisor_invocations=3,
    )

    advisor_called = False

    async def mock_call_fn(target, **kwargs):
        nonlocal advisor_called
        if "run_coder" in target:
            return {
                "complete": True,
                "tests_passed": True,
                "summary": "Typo fixed",
                "files_changed": ["docs/README.md"],
            }
        if "run_issue_advisor" in target:
            advisor_called = True
            raise AssertionError("Issue Advisor should not be called on fast-path")
        raise ValueError(f"Unexpected call: {target}")

    result = await _execute_single_issue(
        issue=issue,
        dag_state=dag_state,
        execute_fn=None,
        config=config,
        call_fn=mock_call_fn,
        node_id="test-node",
        note_fn=None,
        memory_fn=None,
    )

    # Verify fast-path completed without Issue Advisor
    assert result.outcome == IssueOutcome.COMPLETED
    assert result.attempts == 1
    assert not advisor_called, "Issue Advisor should be bypassed on fast-path"
    assert result.advisor_invocations == 0


@pytest.mark.asyncio
async def test_issue_advisor_only_invoked_after_iteration_3():
    """Non-trivial issue should only invoke Issue Advisor after 3+ iterations."""

    issue = PlannedIssue(
        name="complex-feature",
        title="Complex feature",
        description="Multi-iteration feature",
        acceptance_criteria=["Feature works", "Tests pass"],
        guidance=IssueGuidance(trivial=False)
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test-repo",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/artifacts/prd.md",
        architecture_path="/tmp/artifacts/arch.md",
        issues_dir="/tmp/artifacts/issues",
        prd_summary="Feature",
        architecture_summary="Architecture",
        all_issues=[issue],
        levels=[["complex-feature"]],
    )

    config = ExecutionConfig(
        max_coding_iterations=6,
        enable_issue_advisor=True,
        max_advisor_invocations=2,
    )

    iteration_count = 0
    advisor_invoked_at_iteration = None

    async def mock_call_fn(target, **kwargs):
        nonlocal iteration_count, advisor_invoked_at_iteration

        if "run_coder" in target:
            iteration_count += 1
            # Fail first 2 iterations, succeed on 3rd
            if iteration_count < 3:
                return {
                    "complete": True,
                    "tests_passed": False,
                    "summary": f"Iteration {iteration_count} - tests failed",
                    "files_changed": ["feature.py"],
                }
            else:
                return {
                    "complete": True,
                    "tests_passed": False,  # Still fail to trigger advisor
                    "summary": f"Iteration {iteration_count} - still failing",
                    "files_changed": ["feature.py"],
                }

        if "run_code_reviewer" in target:
            return {
                "approved": False,
                "blocking": False,
                "summary": "Needs fixes",
            }

        if "run_issue_advisor" in target:
            advisor_invoked_at_iteration = iteration_count
            # Advisor escalates after seeing failure
            return {
                "action": "ESCALATE_TO_REPLAN",
                "confidence": 0.8,
                "escalation_reason": "Multiple failures, needs replanning",
            }

        raise ValueError(f"Unexpected call: {target}")

    result = await _execute_single_issue(
        issue=issue,
        dag_state=dag_state,
        execute_fn=None,
        config=config,
        call_fn=mock_call_fn,
        node_id="test-node",
        note_fn=None,
        memory_fn=None,
    )

    # Verify Issue Advisor was only invoked AFTER iteration 3
    assert advisor_invoked_at_iteration is not None, "Issue Advisor should have been called"
    assert advisor_invoked_at_iteration >= 3, f"Issue Advisor should only be called after iteration 3, but was called at iteration {advisor_invoked_at_iteration}"
    assert result.outcome == IssueOutcome.FAILED_ESCALATED
    assert result.advisor_invocations == 1


@pytest.mark.asyncio
async def test_advisor_low_confidence_escalation_works_with_timeout_config():
    """Issue Advisor low-confidence escalation should work with timeout configuration."""

    issue = PlannedIssue(
        name="uncertain-task",
        title="Uncertain implementation",
        description="Task with unclear requirements",
        acceptance_criteria=["Something works"],
        guidance=IssueGuidance(trivial=False)
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test-repo",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/artifacts/prd.md",
        architecture_path="/tmp/artifacts/arch.md",
        issues_dir="/tmp/artifacts/issues",
        prd_summary="Unclear requirements",
        architecture_summary="Architecture",
        all_issues=[issue],
        levels=[["uncertain-task"]],
    )

    config = ExecutionConfig(
        max_coding_iterations=6,
        enable_issue_advisor=True,
        max_advisor_invocations=2,
        issue_advisor_timeout=1800,  # Custom timeout from branch 19
    )

    iteration_count = 0

    async def mock_call_fn(target, **kwargs):
        nonlocal iteration_count

        if "run_coder" in target:
            iteration_count += 1
            # Fail multiple times to reach advisor threshold
            return {
                "complete": True,
                "tests_passed": False,
                "summary": f"Iteration {iteration_count} failed",
                "files_changed": ["uncertain.py"],
            }

        if "run_code_reviewer" in target:
            return {
                "approved": False,
                "blocking": False,
                "summary": "Not ready",
            }

        if "run_issue_advisor" in target:
            # Return LOW confidence (< 0.4) to trigger escalation
            return {
                "action": "RETRY_MODIFIED",
                "confidence": 0.3,  # LOW confidence triggers auto-escalation
                "failure_diagnosis": "Requirements unclear",
                "modified_acceptance_criteria": ["Relaxed criteria"],
            }

        raise ValueError(f"Unexpected call: {target}")

    result = await _execute_single_issue(
        issue=issue,
        dag_state=dag_state,
        execute_fn=None,
        config=config,
        call_fn=mock_call_fn,
        node_id="test-node",
        note_fn=None,
        memory_fn=None,
    )

    # Verify low-confidence escalation occurred
    assert result.outcome == IssueOutcome.FAILED_ESCALATED
    assert "low confidence" in result.result_summary.lower() or "0.3" in result.result_summary
    assert result.advisor_invocations >= 1


@pytest.mark.asyncio
async def test_trivial_fast_path_followed_by_normal_issue_with_advisor():
    """Test workflow: trivial fast-path completes, then normal issue uses Issue Advisor."""

    trivial_issue = PlannedIssue(
        name="trivial-setup",
        title="Setup config",
        description="Create initial config file",
        acceptance_criteria=["Config file exists"],
        guidance=IssueGuidance(trivial=True)
    ).model_dump()

    complex_issue = PlannedIssue(
        name="complex-feature",
        title="Complex feature",
        description="Feature requiring multiple attempts",
        acceptance_criteria=["Feature complete"],
        depends_on=["trivial-setup"],
        guidance=IssueGuidance(trivial=False)
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test-repo",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/artifacts/prd.md",
        architecture_path="/tmp/artifacts/arch.md",
        issues_dir="/tmp/artifacts/issues",
        prd_summary="Mixed tasks",
        architecture_summary="Architecture",
        all_issues=[trivial_issue, complex_issue],
        levels=[["trivial-setup"], ["complex-feature"]],
    )

    config = ExecutionConfig(
        max_coding_iterations=6,
        enable_issue_advisor=True,
        max_advisor_invocations=2,
    )

    current_issue_name = None
    advisor_called_for_trivial = False
    advisor_called_for_complex = False
    complex_iteration_count = 0

    async def mock_call_fn(target, **kwargs):
        nonlocal current_issue_name, advisor_called_for_trivial, advisor_called_for_complex, complex_iteration_count

        if "issue" in kwargs:
            current_issue_name = kwargs["issue"]["name"]

        if "run_coder" in target:
            if current_issue_name == "trivial-setup":
                return {
                    "complete": True,
                    "tests_passed": True,
                    "summary": "Config created",
                    "files_changed": ["config.yml"],
                }
            else:  # complex-feature
                complex_iteration_count += 1
                if complex_iteration_count < 4:
                    return {
                        "complete": True,
                        "tests_passed": False,
                        "summary": f"Iteration {complex_iteration_count} failed",
                        "files_changed": ["feature.py"],
                    }
                else:
                    return {
                        "complete": True,
                        "tests_passed": True,
                        "summary": "Feature complete",
                        "files_changed": ["feature.py"],
                    }

        if "run_code_reviewer" in target:
            if current_issue_name == "complex-feature" and complex_iteration_count < 4:
                return {
                    "approved": False,
                    "blocking": False,
                    "summary": "Needs fixes",
                }
            return {
                "approved": True,
                "blocking": False,
                "summary": "Approved",
            }

        if "run_issue_advisor" in target:
            if current_issue_name == "trivial-setup":
                advisor_called_for_trivial = True
            else:
                advisor_called_for_complex = True
            return {
                "action": "RETRY_APPROACH",
                "confidence": 0.7,
                "new_approach": "Try different strategy",
            }

        raise ValueError(f"Unexpected call: {target}")

    # Execute trivial issue
    trivial_result = await _execute_single_issue(
        issue=trivial_issue,
        dag_state=dag_state,
        execute_fn=None,
        config=config,
        call_fn=mock_call_fn,
        node_id="test-node",
        note_fn=None,
        memory_fn=None,
    )

    # Execute complex issue
    complex_result = await _execute_single_issue(
        issue=complex_issue,
        dag_state=dag_state,
        execute_fn=None,
        config=config,
        call_fn=mock_call_fn,
        node_id="test-node",
        note_fn=None,
        memory_fn=None,
    )

    # Verify trivial bypassed advisor, complex used it
    assert trivial_result.outcome == IssueOutcome.COMPLETED
    assert trivial_result.attempts == 1
    assert not advisor_called_for_trivial, "Trivial issue should bypass Issue Advisor"

    assert complex_result.outcome == IssueOutcome.COMPLETED
    assert advisor_called_for_complex, "Complex issue should use Issue Advisor after 3+ iterations"
    assert complex_result.advisor_invocations >= 1
