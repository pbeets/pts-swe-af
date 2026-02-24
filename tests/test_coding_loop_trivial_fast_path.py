"""Tests for trivial issue fast-path in coding loop.

Covers:
- AC1: Trivial flag extracted from guidance at loop start
- AC2: Fast-path condition checked after coder completes: is_trivial and tests_passed and iteration==1
- AC3: Fast-path approval logged with tags=['coding_loop', 'fast_path', 'approve']
- AC4: IssueResult returned with outcome=COMPLETED and attempts=1
- AC5: iteration_history includes fast_path=True flag
- AC6: Non-trivial issues follow standard review path
"""

from __future__ import annotations

import tempfile
import shutil
import pytest

from swe_af.execution.coding_loop import run_coding_loop
from swe_af.execution.schemas import (
    DAGState,
    ExecutionConfig,
    IssueOutcome,
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test isolation."""
    repo_dir = tempfile.mkdtemp(prefix="test-repo-")
    artifacts_dir = tempfile.mkdtemp(prefix="test-artifacts-")
    yield repo_dir, artifacts_dir
    # Cleanup
    shutil.rmtree(repo_dir, ignore_errors=True)
    shutil.rmtree(artifacts_dir, ignore_errors=True)


class MockCallFn:
    """Mock call function that simulates coder and reviewer agents."""

    def __init__(self, coder_response: dict, reviewer_response: dict | None = None):
        self.coder_response = coder_response
        self.reviewer_response = reviewer_response or {"approved": True, "blocking": False, "summary": "LGTM"}
        self.calls = []

    async def __call__(self, method: str, **kwargs):
        """Record calls and return appropriate mock response."""
        self.calls.append({"method": method, "kwargs": kwargs})
        if "run_coder" in method:
            return self.coder_response
        elif "run_code_reviewer" in method:
            return self.reviewer_response
        else:
            return {}


class TestTrivialFastPath:
    """Tests for trivial issue fast-path implementation (AC1-AC5)."""

    @pytest.mark.asyncio
    async def test_trivial_with_tests_passed_completes_in_one_iteration(self, temp_dirs) -> None:
        """Trivial issue with tests_passed=True should complete in 1 iteration (AC2, AC4)."""
        # Setup: trivial issue
        issue = {
            "name": "update-readme",
            "title": "Update README",
            "description": "Add installation instructions to README",
            "acceptance_criteria": ["README contains installation instructions"],
            "depends_on": [],
            "guidance": {
                "trivial": True,
                "needs_deeper_qa": False,
            },
        }

        # Mock coder that reports success with tests passed
        coder_response = {
            "complete": True,
            "tests_passed": True,
            "summary": "Updated README with installation instructions",
            "files_changed": ["README.md"],
        }
        call_fn = MockCallFn(coder_response=coder_response)

        repo_path, artifacts_dir = temp_dirs
        dag_state = DAGState(
            repo_path=repo_path,
            artifacts_dir=artifacts_dir,
        )

        config = ExecutionConfig()

        # Capture notes for validation
        notes = []

        def note_fn(msg: str, tags: list[str] | None = None):
            notes.append({"msg": msg, "tags": tags or []})

        # Execute
        result = await run_coding_loop(
            issue=issue,
            dag_state=dag_state,
            call_fn=call_fn,
            node_id="test-node",
            config=config,
            note_fn=note_fn,
        )

        # AC4: Verify outcome is COMPLETED with attempts=1
        assert result.outcome == IssueOutcome.COMPLETED, "Fast-path should complete successfully"
        assert result.attempts == 1, "Fast-path should complete in 1 iteration"
        assert result.issue_name == "update-readme"
        assert result.files_changed == ["README.md"]

        # AC5: Verify iteration_history includes fast_path=True flag
        assert len(result.iteration_history) == 1, "Should have exactly 1 iteration in history"
        history_entry = result.iteration_history[0]
        assert history_entry["fast_path"] is True, "iteration_history must include fast_path=True"
        assert history_entry["iteration"] == 1
        assert history_entry["action"] == "approve"
        assert history_entry["path"] == "fast_path"

        # AC3: Verify fast-path approval was logged with correct tags
        fast_path_notes = [n for n in notes if "fast_path" in n["tags"] and "approve" in n["tags"]]
        assert len(fast_path_notes) >= 1, "Fast-path approval should be logged"
        fast_path_note = fast_path_notes[0]
        assert "coding_loop" in fast_path_note["tags"], "Should include 'coding_loop' tag"
        assert "fast_path" in fast_path_note["tags"], "Should include 'fast_path' tag"
        assert "approve" in fast_path_note["tags"], "Should include 'approve' tag"
        assert "update-readme" in fast_path_note["tags"], "Should include issue name tag"

        # AC1: Verify trivial flag was logged at start
        trivial_notes = [n for n in notes if "trivial" in n["tags"] and "eligible" in n["tags"]]
        assert len(trivial_notes) >= 1, "Trivial flag should be logged at loop start"

        # Verify reviewer was NOT called (fast-path skips review)
        reviewer_calls = [c for c in call_fn.calls if "run_code_reviewer" in c["method"]]
        assert len(reviewer_calls) == 0, "Fast-path should skip reviewer"

    @pytest.mark.asyncio
    async def test_trivial_with_tests_failed_continues_to_review(self, temp_dirs) -> None:
        """Trivial issue with tests_passed=False should continue to review (AC2, AC6)."""
        # Setup: trivial issue
        issue = {
            "name": "update-config",
            "title": "Update Config",
            "description": "Change timeout value",
            "acceptance_criteria": ["Config timeout is 30s"],
            "depends_on": [],
            "guidance": {
                "trivial": True,
                "needs_deeper_qa": False,
            },
        }

        # Mock call_fn that returns different results per iteration
        call_count = 0

        async def call_fn(method: str, **kwargs):
            nonlocal call_count
            if "run_coder" in method:
                call_count += 1
                if call_count == 1:
                    # First iteration: tests fail
                    return {
                        "complete": True,
                        "tests_passed": False,
                        "summary": "Updated config but tests failed",
                        "files_changed": ["config.yaml"],
                    }
                else:
                    # Second iteration: tests pass
                    return {
                        "complete": True,
                        "tests_passed": True,
                        "summary": "Updated config, tests passing",
                        "files_changed": ["config.yaml"],
                    }
            elif "run_code_reviewer" in method:
                return {
                    "approved": True,
                    "blocking": False,
                    "summary": "Config looks good after test fixes",
                }
            return {}

        repo_path, artifacts_dir = temp_dirs
        dag_state = DAGState(
            repo_path=repo_path,
            artifacts_dir=artifacts_dir,
        )

        config = ExecutionConfig()

        notes = []

        def note_fn(msg: str, tags: list[str] | None = None):
            notes.append({"msg": msg, "tags": tags or []})

        # Execute
        result = await run_coding_loop(
            issue=issue,
            dag_state=dag_state,
            call_fn=call_fn,
            node_id="test-node",
            config=config,
            note_fn=note_fn,
        )

        # AC6: Verify that standard review path was followed
        assert result.outcome == IssueOutcome.COMPLETED, "Should complete via standard path"
        # Should take 2 iterations: first fails tests, second passes
        assert result.attempts == 2, "Should complete in 2 iterations when tests fail initially"

        # Verify fast-path was NOT used (no fast_path in any history entry)
        for entry in result.iteration_history:
            assert entry.get("fast_path") is None or entry.get("fast_path") is False, \
                "fast_path should not be set when tests fail"

        # Verify fast-path approval was NOT logged
        fast_path_notes = [n for n in notes if "fast_path" in n["tags"] and "approve" in n["tags"]]
        assert len(fast_path_notes) == 0, "Fast-path approval should not be logged when tests fail"

    @pytest.mark.asyncio
    async def test_non_trivial_issue_follows_standard_path(self, temp_dirs) -> None:
        """Non-trivial issue should follow standard review path (AC6)."""
        # Setup: non-trivial issue (trivial=False)
        issue = {
            "name": "implement-feature",
            "title": "Implement New Feature",
            "description": "Add complex business logic",
            "acceptance_criteria": [
                "Feature implemented",
                "Tests passing",
                "Documentation updated",
            ],
            "depends_on": ["dependency-issue"],
            "guidance": {
                "trivial": False,
                "needs_deeper_qa": False,
            },
        }

        # Mock coder that reports success with tests passed
        coder_response = {
            "complete": True,
            "tests_passed": True,
            "summary": "Implemented feature successfully",
            "files_changed": ["src/feature.py", "tests/test_feature.py"],
        }

        # Mock reviewer that approves
        reviewer_response = {
            "approved": True,
            "blocking": False,
            "summary": "Implementation looks good",
        }

        call_fn = MockCallFn(coder_response=coder_response, reviewer_response=reviewer_response)

        repo_path, artifacts_dir = temp_dirs
        dag_state = DAGState(
            repo_path=repo_path,
            artifacts_dir=artifacts_dir,
        )

        config = ExecutionConfig()

        notes = []

        def note_fn(msg: str, tags: list[str] | None = None):
            notes.append({"msg": msg, "tags": tags or []})

        # Execute
        result = await run_coding_loop(
            issue=issue,
            dag_state=dag_state,
            call_fn=call_fn,
            node_id="test-node",
            config=config,
            note_fn=note_fn,
        )

        # AC6: Verify standard path was followed
        assert result.outcome == IssueOutcome.COMPLETED, "Should complete successfully"
        assert result.attempts >= 1, "Should complete successfully"

        # Verify fast-path was NOT used in any history entry
        for entry in result.iteration_history:
            assert entry.get("fast_path") is None or entry.get("fast_path") is False, \
                "Non-trivial issues should not use fast-path"

        # Verify reviewer WAS called
        reviewer_calls = [c for c in call_fn.calls if "run_code_reviewer" in c["method"]]
        assert len(reviewer_calls) >= 1, "Non-trivial issues should call reviewer"

        # Verify fast-path approval was NOT logged
        fast_path_notes = [n for n in notes if "fast_path" in n["tags"] and "approve" in n["tags"]]
        assert len(fast_path_notes) == 0, "Fast-path approval should not be logged for non-trivial issues"

        # Verify trivial flag was NOT logged at start
        trivial_notes = [n for n in notes if "trivial" in n["tags"] and "eligible" in n["tags"]]
        assert len(trivial_notes) == 0, "Trivial flag should not be logged for non-trivial issues"

    @pytest.mark.asyncio
    async def test_trivial_on_second_iteration_no_fast_path(self, temp_dirs) -> None:
        """Trivial issue on iteration 2 should not use fast-path (AC2)."""
        # Setup: trivial issue that will fail first iteration
        issue = {
            "name": "update-docs",
            "title": "Update Documentation",
            "description": "Fix typo in docs",
            "acceptance_criteria": ["Typo fixed"],
            "depends_on": [],
            "guidance": {
                "trivial": True,
                "needs_deeper_qa": False,
            },
        }

        call_count = {"coder": 0, "reviewer": 0}

        async def call_fn_multi_iter(method: str, **kwargs):
            """Mock that fails first iteration, succeeds on second."""
            if "run_coder" in method:
                call_count["coder"] += 1
                if call_count["coder"] == 1:
                    # First iteration: tests fail
                    return {
                        "complete": True,
                        "tests_passed": False,
                        "summary": "Fixed typo but tests failed",
                        "files_changed": ["docs/README.md"],
                    }
                else:
                    # Second iteration: tests pass
                    return {
                        "complete": True,
                        "tests_passed": True,
                        "summary": "Fixed typo and tests pass",
                        "files_changed": ["docs/README.md"],
                    }
            elif "run_code_reviewer" in method:
                call_count["reviewer"] += 1
                if call_count["reviewer"] == 1:
                    # First review: request fix (not approved)
                    return {
                        "approved": False,
                        "blocking": False,
                        "summary": "Please fix test failures",
                    }
                else:
                    # Second review: approve
                    return {
                        "approved": True,
                        "blocking": False,
                        "summary": "LGTM",
                    }
            return {}

        repo_path, artifacts_dir = temp_dirs
        dag_state = DAGState(
            repo_path=repo_path,
            artifacts_dir=artifacts_dir,
        )

        config = ExecutionConfig()

        notes = []

        def note_fn(msg: str, tags: list[str] | None = None):
            notes.append({"msg": msg, "tags": tags or []})

        # Execute
        result = await run_coding_loop(
            issue=issue,
            dag_state=dag_state,
            call_fn=call_fn_multi_iter,
            node_id="test-node",
            config=config,
            note_fn=note_fn,
        )

        # AC2: Verify fast-path was NOT used (iteration != 1)
        assert result.outcome == IssueOutcome.COMPLETED, "Should complete successfully"
        # Due to the mock behavior, it may take multiple iterations, but that's okay
        # The key is that fast-path was NOT used
        assert result.attempts >= 2, f"Should complete in 2+ iterations, got {result.attempts}"

        # Verify no fast_path in any iteration history
        for entry in result.iteration_history:
            assert entry.get("fast_path") is None or entry.get("fast_path") is False, \
                "Fast-path should not be used on iteration 2+"

        # Verify fast-path approval was NOT logged
        fast_path_notes = [n for n in notes if "fast_path" in n["tags"] and "approve" in n["tags"]]
        assert len(fast_path_notes) == 0, "Fast-path approval should not be logged on iteration 2+"

        # Verify that reviewer WAS called (fast-path not taken)
        assert call_count["reviewer"] > 0, "Reviewer should be called when fast-path is not taken"


class TestFastPathConditions:
    """Tests for fast-path condition combinations."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("trivial,tests_passed,iteration,expected_fast_path", [
        # AC2: All conditions must be True for fast-path
        (True, True, 1, True),      # All conditions met -> fast-path
        (True, False, 1, False),    # Tests failed -> no fast-path
        (False, True, 1, False),    # Not trivial -> no fast-path
        (True, True, 2, False),     # Not iteration 1 -> no fast-path
        (False, False, 1, False),   # Multiple conditions false -> no fast-path
    ])
    async def test_fast_path_condition_combinations(
        self,
        temp_dirs,
        trivial: bool,
        tests_passed: bool,
        iteration: int,
        expected_fast_path: bool,
    ) -> None:
        """Test all combinations of fast-path conditions (AC2)."""
        issue = {
            "name": "test-issue",
            "title": "Test Issue",
            "description": "Test",
            "acceptance_criteria": ["AC1"],
            "depends_on": [],
            "guidance": {
                "trivial": trivial,
                "needs_deeper_qa": False,
            },
        }

        coder_response = {
            "complete": True,
            "tests_passed": tests_passed,
            "summary": "Work done",
            "files_changed": ["test.txt"],
        }

        reviewer_response = {
            "approved": True,
            "blocking": False,
            "summary": "LGTM",
        }

        call_fn = MockCallFn(coder_response=coder_response, reviewer_response=reviewer_response)

        repo_path, artifacts_dir = temp_dirs
        dag_state = DAGState(
            repo_path=repo_path,
            artifacts_dir=artifacts_dir,
        )

        config = ExecutionConfig()

        notes = []

        def note_fn(msg: str, tags: list[str] | None = None):
            notes.append({"msg": msg, "tags": tags or []})

        # For iteration > 1, we need to simulate previous iterations
        if iteration > 1:
            # This test case would require more complex setup
            # For now, we only test iteration 1
            pytest.skip("Multi-iteration test requires complex setup")

        # Execute
        result = await run_coding_loop(
            issue=issue,
            dag_state=dag_state,
            call_fn=call_fn,
            node_id="test-node",
            config=config,
            note_fn=note_fn,
        )

        # Verify fast-path was used or not based on expected_fast_path
        fast_path_notes = [n for n in notes if "fast_path" in n["tags"] and "approve" in n["tags"]]

        if expected_fast_path:
            assert len(fast_path_notes) >= 1, f"Fast-path should be used when all conditions are met. Notes: {notes}"
            assert result.iteration_history[0].get("fast_path") is True, f"History should show fast_path=True. History: {result.iteration_history}"
        else:
            assert len(fast_path_notes) == 0, f"Fast-path should not be used when conditions not met (trivial={trivial}, tests={tests_passed}, iter={iteration})"
            # If completed, history should not have fast_path=True in any entry
            if result.outcome == IssueOutcome.COMPLETED and result.iteration_history:
                for entry in result.iteration_history:
                    assert entry.get("fast_path") is None or entry.get("fast_path") is False, \
                           f"History should not show fast_path=True when conditions not met. Entry: {entry}"
