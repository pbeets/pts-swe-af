"""Integration test: Issue Advisor iteration threshold + confidence escalation.

Tests the interaction between:
1. Issue Advisor gating logic (only invoked after 3+ iterations)
2. Low-confidence escalation (confidence < 0.4 → FAILED_ESCALATED)
3. Coding loop iteration accumulation

These features from issue-19 and issue-14 work together in dag_executor.
"""

import pytest
from swe_af.execution.schemas import ExecutionConfig, DAGState, IssueOutcome
from swe_af.execution.dag_executor import _execute_single_issue
from swe_af.reasoners.schemas import PlannedIssue, IssueGuidance


@pytest.mark.asyncio
async def test_advisor_not_invoked_below_iteration_threshold():
    """Issue Advisor should NOT be invoked if coding loop completes in < 3 iterations."""

    config = ExecutionConfig(
        max_coding_iterations=6,
        max_advisor_invocations=2,
        enable_issue_advisor=True,
    )

    issue = PlannedIssue(
        name="test-issue",
        title="Test Issue",
        description="Test",
        acceptance_criteria=["AC1"],
        guidance=IssueGuidance(trivial=False)
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/prd.md",
        architecture_path="/tmp/arch.md",
        issues_dir="/tmp/issues",
        prd_summary="Test",
        architecture_summary="Test",
        all_issues=[issue],
        levels=[["test-issue"]],
    )

    call_counts = {"coder": 0, "reviewer": 0, "advisor": 0}

    async def mock_call_fn(target, **kwargs):
        if "run_coder" in target:
            call_counts["coder"] += 1
            # Fail first 2 iterations, succeed on 3rd
            if call_counts["coder"] < 2:
                return {
                    "complete": False,
                    "tests_passed": False,
                    "summary": "Still working",
                    "files_changed": ["file.py"],
                }
            else:
                return {
                    "complete": True,
                    "tests_passed": True,
                    "summary": "Done",
                    "files_changed": ["file.py"],
                }
        if "run_code_reviewer" in target:
            call_counts["reviewer"] += 1
            if call_counts["reviewer"] < 2:
                return {
                    "approved": False,
                    "blocking": False,
                    "summary": "Needs fixes",
                }
            else:
                return {
                    "approved": True,
                    "blocking": False,
                    "summary": "LGTM",
                }
        if "run_issue_advisor" in target:
            call_counts["advisor"] += 1
            # Should not be reached
            return {
                "action": "accept_with_debt",
                "confidence": 0.8,
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

    # Should complete after 2 iterations (< 3 threshold)
    assert result.outcome == IssueOutcome.COMPLETED
    assert result.attempts == 2
    assert call_counts["advisor"] == 0, "Advisor should NOT be invoked (< 3 iterations)"


@pytest.mark.asyncio
async def test_advisor_invoked_after_threshold_with_failure():
    """Issue Advisor SHOULD be invoked when coding loop fails after 3+ iterations."""

    config = ExecutionConfig(
        max_coding_iterations=6,
        max_advisor_invocations=1,
        enable_issue_advisor=True,
    )

    issue = PlannedIssue(
        name="test-issue",
        title="Test Issue",
        description="Test",
        acceptance_criteria=["AC1"],
        guidance=IssueGuidance(trivial=False)
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/prd.md",
        architecture_path="/tmp/arch.md",
        issues_dir="/tmp/issues",
        prd_summary="Test",
        architecture_summary="Test",
        all_issues=[issue],
        levels=[["test-issue"]],
    )

    call_counts = {"coder": 0, "reviewer": 0, "advisor": 0}

    async def mock_call_fn(target, **kwargs):
        if "run_coder" in target:
            call_counts["coder"] += 1
            # Always fail to reach 3+ iterations
            return {
                "complete": False,
                "tests_passed": False,
                "summary": f"Failed iteration {call_counts['coder']}",
                "files_changed": ["file.py"],
            }
        if "run_code_reviewer" in target:
            call_counts["reviewer"] += 1
            return {
                "approved": False,
                "blocking": True,
                "summary": "Blocking issues",
            }
        if "run_issue_advisor" in target:
            call_counts["advisor"] += 1
            iteration_count = kwargs.get("failure_result", {}).get("attempts", 0)
            return {
                "action": "accept_with_debt",
                "confidence": 0.8,
                "summary": f"Accept with debt after {iteration_count} iterations",
                "missing_functionality": ["Some features incomplete"],
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

    # Advisor should be invoked after 3 iterations
    assert call_counts["coder"] >= 3, "Should reach at least 3 coding iterations"
    assert call_counts["advisor"] == 1, "Advisor should be invoked once after threshold"
    assert result.outcome == IssueOutcome.COMPLETED_WITH_DEBT
    assert result.advisor_invocations == 1


@pytest.mark.asyncio
async def test_low_confidence_advisor_triggers_escalation():
    """Low confidence (< 0.4) from Issue Advisor should escalate to replanner."""

    config = ExecutionConfig(
        max_coding_iterations=6,
        max_advisor_invocations=1,
        enable_issue_advisor=True,
    )

    issue = PlannedIssue(
        name="complex-issue",
        title="Complex Issue",
        description="Very complex implementation",
        acceptance_criteria=["AC1", "AC2", "AC3"],
        guidance=IssueGuidance(trivial=False)
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/prd.md",
        architecture_path="/tmp/arch.md",
        issues_dir="/tmp/issues",
        prd_summary="Test",
        architecture_summary="Test",
        all_issues=[issue],
        levels=[["complex-issue"]],
    )

    call_counts = {"coder": 0, "advisor": 0}

    async def mock_call_fn(target, **kwargs):
        if "run_coder" in target:
            call_counts["coder"] += 1
            return {
                "complete": False,
                "tests_passed": False,
                "summary": "Complex failure",
                "files_changed": [],
            }
        if "run_code_reviewer" in target:
            return {
                "approved": False,
                "blocking": True,
                "summary": "Major architectural issues",
            }
        if "run_issue_advisor" in target:
            call_counts["advisor"] += 1
            # Return low confidence to trigger escalation
            return {
                "action": "retry_approach",
                "confidence": 0.25,  # Below 0.4 threshold
                "failure_diagnosis": "Issue is too complex for simple fixes",
                "escalation_reason": "Requires architectural changes",
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

    # Should escalate due to low confidence
    assert result.outcome == IssueOutcome.FAILED_ESCALATED
    assert call_counts["advisor"] == 1
    assert "Low advisor confidence" in result.result_summary or "low confidence" in result.result_summary.lower()
    assert result.escalation_context is not None
    assert "0.25" in result.escalation_context or "confidence" in result.escalation_context.lower()


@pytest.mark.asyncio
async def test_high_confidence_advisor_no_escalation():
    """High confidence (≥ 0.4) from Issue Advisor should NOT escalate."""

    config = ExecutionConfig(
        max_coding_iterations=6,
        max_advisor_invocations=1,
        enable_issue_advisor=True,
    )

    issue = PlannedIssue(
        name="test-issue",
        title="Test Issue",
        description="Test",
        acceptance_criteria=["AC1"],
        guidance=IssueGuidance(trivial=False)
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/prd.md",
        architecture_path="/tmp/arch.md",
        issues_dir="/tmp/issues",
        prd_summary="Test",
        architecture_summary="Test",
        all_issues=[issue],
        levels=[["test-issue"]],
    )

    async def mock_call_fn(target, **kwargs):
        if "run_coder" in target:
            return {
                "complete": False,
                "tests_passed": False,
                "summary": "Failed",
                "files_changed": ["file.py"],
            }
        if "run_code_reviewer" in target:
            return {
                "approved": False,
                "blocking": True,
                "summary": "Needs fixes",
            }
        if "run_issue_advisor" in target:
            # Return acceptable confidence (above threshold)
            return {
                "action": "accept_with_debt",
                "confidence": 0.75,  # Well above 0.4 threshold
                "summary": "Accept with minor debt",
                "missing_functionality": ["Minor feature gap"],
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

    # Should NOT escalate
    assert result.outcome == IssueOutcome.COMPLETED_WITH_DEBT
    assert result.advisor_invocations == 1
    # Should not have escalation context
    assert result.escalation_context is None or result.escalation_context == ""


@pytest.mark.asyncio
async def test_iteration_accumulation_across_advisor_rounds():
    """Verify iteration count accumulates correctly across multiple advisor rounds."""

    config = ExecutionConfig(
        max_coding_iterations=3,  # 3 per coding loop attempt
        max_advisor_invocations=2,
        enable_issue_advisor=True,
    )

    issue = PlannedIssue(
        name="test-issue",
        title="Test Issue",
        description="Test",
        acceptance_criteria=["AC1"],
        guidance=IssueGuidance(trivial=False)
    ).model_dump()

    dag_state = DAGState(
        repo_path="/tmp/test",
        artifacts_dir="/tmp/artifacts",
        prd_path="/tmp/prd.md",
        architecture_path="/tmp/arch.md",
        issues_dir="/tmp/issues",
        prd_summary="Test",
        architecture_summary="Test",
        all_issues=[issue],
        levels=[["test-issue"]],
    )

    advisor_calls = []

    async def mock_call_fn(target, **kwargs):
        if "run_coder" in target:
            # Always fail to exhaust iterations
            return {
                "complete": False,
                "tests_passed": False,
                "summary": "Failed",
                "files_changed": [],
            }
        if "run_code_reviewer" in target:
            return {
                "approved": False,
                "blocking": True,
                "summary": "Not approved",
            }
        if "run_issue_advisor" in target:
            # Track cumulative iteration count
            failure_result = kwargs.get("failure_result", {})
            attempts = failure_result.get("attempts", 0)
            advisor_calls.append(attempts)
            # First advisor call: retry with approach
            if len(advisor_calls) == 1:
                return {
                    "action": "retry_approach",
                    "confidence": 0.8,
                    "new_approach": "Try different approach",
                }
            else:
                # Second call: accept with debt
                return {
                    "action": "accept_with_debt",
                    "confidence": 0.7,
                    "missing_functionality": [],
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

    # Verify iteration accumulation
    assert len(advisor_calls) >= 1, "Advisor should be called at least once"
    # First advisor call should see 3 iterations (max_coding_iterations)
    assert advisor_calls[0] == 3, f"First advisor should see 3 iterations, got {advisor_calls[0]}"

    if len(advisor_calls) == 2:
        # Second advisor call should see 6 iterations (3 + 3)
        assert advisor_calls[1] == 6, f"Second advisor should see accumulated 6 iterations, got {advisor_calls[1]}"
