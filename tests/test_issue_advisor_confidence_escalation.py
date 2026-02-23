"""Unit tests for Issue Advisor low-confidence escalation.

Tests verify that:
1. Confidence is extracted from advisor_decision.get('confidence', 0.5)
2. Low-confidence check: confidence < 0.4 triggers escalation
3. Escalation logged with tags=['issue_advisor', 'escalate', 'low_confidence']
4. IssueResult returned with outcome=FAILED_ESCALATED
5. escalation_context includes confidence score
6. Default confidence 0.5 prevents spurious escalations on missing field
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from swe_af.execution.dag_executor import _execute_single_issue
from swe_af.execution.schemas import DAGState, ExecutionConfig, IssueOutcome, IssueResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dag_state(artifacts_dir: str, repo_path: str = "/tmp/fake-repo") -> DAGState:
    """Create a minimal DAGState for testing."""
    return DAGState(
        repo_path=repo_path,
        artifacts_dir=artifacts_dir,
        prd_path="",
        architecture_path="",
        issues_dir="",
    )


def _make_config(**overrides) -> ExecutionConfig:
    """Create an ExecutionConfig with sensible test defaults."""
    defaults = {
        "max_coding_iterations": 6,
        "agent_timeout_seconds": 30,
        "max_advisor_invocations": 2,
        "enable_issue_advisor": True,
    }
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def _make_issue(name: str = "ISSUE-1", **extra) -> dict:
    """Create a minimal issue dict."""
    issue = {
        "name": name,
        "title": "Test issue",
        "description": "A test issue for confidence escalation",
        "acceptance_criteria": ["AC-1: it works"],
        "depends_on": [],
        "provides": [],
        "files_to_create": [],
        "files_to_modify": [],
        "worktree_path": "/tmp/fake-repo",
        "branch_name": "test/issue-1",
    }
    issue.update(extra)
    return issue


class _MockCallFn:
    """Mock call_fn that tracks invocations and returns scripted responses."""

    def __init__(self, advisor_confidence: float = 0.5):
        self.calls = []
        self.advisor_confidence = advisor_confidence

    async def __call__(self, method: str, **kwargs):
        """Record call and return scripted response based on method."""
        self.calls.append({"method": method, "kwargs": kwargs})

        if "run_issue_advisor" in method:
            # Return advisor decision with specified confidence
            return {
                "action": "retry_approach",
                "new_approach": "Try a different approach",
                "approach_changes": ["Change X", "Change Y"],
                "failure_diagnosis": "Low confidence diagnosis",
                "rationale": "Test rationale",
                "downstream_impact": "None",
                "confidence": self.advisor_confidence,
                "escalation_reason": "Unable to provide confident guidance",
            }

        # Default for other methods
        return {}

    def get_advisor_invocations(self) -> int:
        """Count how many times the Issue Advisor was invoked."""
        return sum(1 for call in self.calls if "run_issue_advisor" in call["method"])


class _NoteCollector:
    """Collect note_fn calls for test assertions."""

    def __init__(self):
        self.notes = []

    def __call__(self, message: str, tags: list[str] | None = None):
        self.notes.append({"message": message, "tags": tags or []})

    def has_low_confidence_escalation_note(self) -> bool:
        """Check if low-confidence escalation note was emitted."""
        return any(
            "issue_advisor" in note["tags"]
            and "escalate" in note["tags"]
            and "low_confidence" in note["tags"]
            for note in self.notes
        )

    def get_escalation_notes(self) -> list[dict]:
        """Get all escalation notes."""
        return [
            note
            for note in self.notes
            if "escalate" in note["tags"] and "low_confidence" in note["tags"]
        ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIssueAdvisorConfidenceEscalation(unittest.TestCase):
    """Test suite for Issue Advisor low-confidence escalation."""

    def setUp(self):
        """Create temp directory for test artifacts."""
        self.test_dir = tempfile.mkdtemp()
        self.artifacts_dir = os.path.join(self.test_dir, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_low_confidence_triggers_escalation(self):
        """AC1-AC5: Confidence=0.3 (<0.4) triggers FAILED_ESCALATED outcome."""
        dag_state = _make_dag_state(self.artifacts_dir)
        config = _make_config()
        issue = _make_issue()

        # Mock call_fn with low confidence (0.3)
        mock_call_fn = _MockCallFn(advisor_confidence=0.3)
        note_collector = _NoteCollector()

        async def run_test():
            # Mock run_coding_loop to return failures with 3+ attempts
            async def mock_coding_loop(**kwargs):
                return IssueResult(
                    issue_name=issue["name"],
                    outcome=IssueOutcome.FAILED_RETRYABLE,
                    attempts=3,
                    result_summary="Failed after 3 iterations",
                    error_message="Test error",
                    error_context="Test context",
                    files_changed=[],
                    iteration_history=[{"iteration": i} for i in range(1, 4)],
                )

            with patch('swe_af.execution.coding_loop.run_coding_loop', new=mock_coding_loop):
                result = await _execute_single_issue(
                    issue=issue,
                    dag_state=dag_state,
                    execute_fn=None,
                    config=config,
                    call_fn=mock_call_fn,
                    note_fn=note_collector,
                )
            return result

        result = asyncio.run(run_test())

        # AC2: Check low-confidence detection (confidence < 0.4)
        self.assertEqual(mock_call_fn.get_advisor_invocations(), 1,
                         "Advisor should be invoked once")

        # AC3: Verify escalation logged with correct tags
        self.assertTrue(note_collector.has_low_confidence_escalation_note(),
                        "Low-confidence escalation note should be emitted")

        escalation_notes = note_collector.get_escalation_notes()
        self.assertGreater(len(escalation_notes), 0, "Should have escalation notes")

        # Verify tags include all required values
        first_note = escalation_notes[0]
        self.assertIn("issue_advisor", first_note["tags"])
        self.assertIn("escalate", first_note["tags"])
        self.assertIn("low_confidence", first_note["tags"])

        # AC4: Verify outcome=FAILED_ESCALATED
        self.assertEqual(result.outcome, IssueOutcome.FAILED_ESCALATED,
                         "Outcome should be FAILED_ESCALATED")

        # AC5: Verify escalation_context includes confidence score
        self.assertIn("0.3", result.escalation_context,
                      "Escalation context should include confidence score")
        self.assertIn("confidence", result.escalation_context.lower(),
                      "Escalation context should mention confidence")

    def test_normal_confidence_proceeds_with_retry(self):
        """AC6: Confidence=0.5 (default) allows normal retry flow."""
        dag_state = _make_dag_state(self.artifacts_dir)
        config = _make_config()
        issue = _make_issue()

        # Mock call_fn with normal confidence (0.5)
        mock_call_fn = _MockCallFn(advisor_confidence=0.5)
        note_collector = _NoteCollector()

        # Track how many times coding loop is called
        coding_loop_call_count = 0

        async def run_test():
            nonlocal coding_loop_call_count

            # Mock run_coding_loop to return failures, then success
            async def mock_coding_loop(**kwargs):
                nonlocal coding_loop_call_count
                coding_loop_call_count += 1

                if coding_loop_call_count == 1:
                    # First call: fail with 3 attempts to trigger advisor
                    return IssueResult(
                        issue_name=issue["name"],
                        outcome=IssueOutcome.FAILED_RETRYABLE,
                        attempts=3,
                        result_summary="Failed after 3 iterations",
                        error_message="Test error",
                        files_changed=[],
                        iteration_history=[{"iteration": i} for i in range(1, 4)],
                    )
                else:
                    # Second call after advisor: succeed
                    return IssueResult(
                        issue_name=issue["name"],
                        outcome=IssueOutcome.COMPLETED,
                        attempts=1,
                        result_summary="Completed after retry",
                        files_changed=["file.py"],
                        iteration_history=[{"iteration": 1}],
                    )

            with patch('swe_af.execution.coding_loop.run_coding_loop', new=mock_coding_loop):
                result = await _execute_single_issue(
                    issue=issue,
                    dag_state=dag_state,
                    execute_fn=None,
                    config=config,
                    call_fn=mock_call_fn,
                    note_fn=note_collector,
                )
            return result

        result = asyncio.run(run_test())

        # Verify: advisor was invoked but did NOT escalate
        self.assertEqual(mock_call_fn.get_advisor_invocations(), 1,
                         "Advisor should be invoked once")
        self.assertFalse(note_collector.has_low_confidence_escalation_note(),
                         "No low-confidence escalation should occur")

        # Verify: retry proceeded normally and completed
        self.assertEqual(result.outcome, IssueOutcome.COMPLETED,
                         "Outcome should be COMPLETED after successful retry")
        self.assertEqual(coding_loop_call_count, 2,
                         "Coding loop should be called twice (initial + retry)")

    def test_missing_confidence_field_defaults_to_0_5(self):
        """AC1,AC6: Missing confidence field defaults to 0.5, no escalation."""
        dag_state = _make_dag_state(self.artifacts_dir)
        config = _make_config()
        issue = _make_issue()

        note_collector = _NoteCollector()

        # Custom mock that omits confidence field entirely
        class _MockCallFnNoConfidence:
            def __init__(self):
                self.calls = []

            async def __call__(self, method: str, **kwargs):
                self.calls.append({"method": method, "kwargs": kwargs})

                if "run_issue_advisor" in method:
                    # Return advisor decision WITHOUT confidence field
                    return {
                        "action": "accept_with_debt",
                        "missing_functionality": ["Feature X"],
                        "debt_severity": "medium",
                        "failure_diagnosis": "Test diagnosis",
                        "rationale": "Test rationale",
                        "downstream_impact": "None",
                        "summary": "Accepted with debt",
                        # Note: confidence field is MISSING
                    }

                return {}

            def get_advisor_invocations(self):
                return sum(1 for call in self.calls if "run_issue_advisor" in call["method"])

        mock_call_fn = _MockCallFnNoConfidence()

        async def run_test():
            # Mock run_coding_loop to return failures
            async def mock_coding_loop(**kwargs):
                return IssueResult(
                    issue_name=issue["name"],
                    outcome=IssueOutcome.FAILED_RETRYABLE,
                    attempts=3,
                    result_summary="Failed after 3 iterations",
                    files_changed=[],
                    iteration_history=[{"iteration": i} for i in range(1, 4)],
                )

            with patch('swe_af.execution.coding_loop.run_coding_loop', new=mock_coding_loop):
                result = await _execute_single_issue(
                    issue=issue,
                    dag_state=dag_state,
                    execute_fn=None,
                    config=config,
                    call_fn=mock_call_fn,
                    note_fn=note_collector,
                )
            return result

        result = asyncio.run(run_test())

        # AC6: Verify no escalation occurred (default 0.5 > 0.4)
        self.assertFalse(note_collector.has_low_confidence_escalation_note(),
                         "No low-confidence escalation should occur with missing field")

        # Verify: advisor processed normally (ACCEPT_WITH_DEBT)
        self.assertEqual(result.outcome, IssueOutcome.COMPLETED_WITH_DEBT,
                         "Outcome should be COMPLETED_WITH_DEBT (advisor's action)")

    def test_boundary_confidence_0_4_does_not_escalate(self):
        """Boundary test: confidence=0.4 should NOT escalate (< 0.4 only)."""
        dag_state = _make_dag_state(self.artifacts_dir)
        config = _make_config()
        issue = _make_issue()

        # Mock call_fn with boundary confidence (0.4)
        mock_call_fn = _MockCallFn(advisor_confidence=0.4)
        note_collector = _NoteCollector()

        coding_loop_call_count = 0

        async def run_test():
            nonlocal coding_loop_call_count

            async def mock_coding_loop(**kwargs):
                nonlocal coding_loop_call_count
                coding_loop_call_count += 1

                if coding_loop_call_count == 1:
                    return IssueResult(
                        issue_name=issue["name"],
                        outcome=IssueOutcome.FAILED_RETRYABLE,
                        attempts=3,
                        result_summary="Failed",
                        files_changed=[],
                        iteration_history=[{"iteration": i} for i in range(1, 4)],
                    )
                else:
                    return IssueResult(
                        issue_name=issue["name"],
                        outcome=IssueOutcome.COMPLETED,
                        attempts=1,
                        result_summary="Completed",
                        files_changed=["file.py"],
                        iteration_history=[{"iteration": 1}],
                    )

            with patch('swe_af.execution.coding_loop.run_coding_loop', new=mock_coding_loop):
                result = await _execute_single_issue(
                    issue=issue,
                    dag_state=dag_state,
                    execute_fn=None,
                    config=config,
                    call_fn=mock_call_fn,
                    note_fn=note_collector,
                )
            return result

        result = asyncio.run(run_test())

        # Verify: no escalation at boundary (0.4 is NOT < 0.4)
        self.assertFalse(note_collector.has_low_confidence_escalation_note(),
                         "No escalation should occur at boundary confidence=0.4")

        # Verify: retry proceeded normally
        self.assertEqual(result.outcome, IssueOutcome.COMPLETED,
                         "Outcome should be COMPLETED after successful retry")

    def test_very_low_confidence_0_1_escalates(self):
        """Edge case: very low confidence (0.1) should escalate."""
        dag_state = _make_dag_state(self.artifacts_dir)
        config = _make_config()
        issue = _make_issue()

        mock_call_fn = _MockCallFn(advisor_confidence=0.1)
        note_collector = _NoteCollector()

        async def run_test():
            async def mock_coding_loop(**kwargs):
                return IssueResult(
                    issue_name=issue["name"],
                    outcome=IssueOutcome.FAILED_RETRYABLE,
                    attempts=3,
                    result_summary="Failed",
                    files_changed=[],
                    iteration_history=[{"iteration": i} for i in range(1, 4)],
                )

            with patch('swe_af.execution.coding_loop.run_coding_loop', new=mock_coding_loop):
                result = await _execute_single_issue(
                    issue=issue,
                    dag_state=dag_state,
                    execute_fn=None,
                    config=config,
                    call_fn=mock_call_fn,
                    note_fn=note_collector,
                )
            return result

        result = asyncio.run(run_test())

        # Verify: very low confidence triggers escalation
        self.assertTrue(note_collector.has_low_confidence_escalation_note(),
                        "Very low confidence (0.1) should trigger escalation")
        self.assertEqual(result.outcome, IssueOutcome.FAILED_ESCALATED,
                         "Outcome should be FAILED_ESCALATED")
        self.assertIn("0.1", result.escalation_context,
                      "Escalation context should include confidence score 0.1")


if __name__ == "__main__":
    unittest.main()
