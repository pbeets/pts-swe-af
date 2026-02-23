"""Test suite for coder.py prompt compression.

Validates:
- AC1: coder.py reduced from 235 to ≤188 LOC (20% reduction)
- AC2: Acceptance criteria enforcement logic preserved
- AC3: Test execution requirements maintained
- AC4: Tool usage instructions remain complete
- AC5: Git operation guidelines preserved
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


# Direct import to avoid circular import issues
def _load_coder_module():
    """Load coder.py module directly without triggering circular imports."""
    coder_path = Path(__file__).parent.parent / "swe_af" / "prompts" / "coder.py"
    spec = importlib.util.spec_from_file_location("coder", coder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

coder = _load_coder_module()
SYSTEM_PROMPT = coder.SYSTEM_PROMPT
coder_task_prompt = coder.coder_task_prompt


def test_coder_py_loc_reduction():
    """AC1: coder.py reduced from 235 to ≤188 LOC (20% reduction)."""
    coder_py = Path(__file__).parent.parent / "swe_af" / "prompts" / "coder.py"
    result = subprocess.run(
        ["wc", "-l", str(coder_py)],
        capture_output=True,
        text=True,
        check=True,
    )
    loc = int(result.stdout.strip().split()[0])
    assert loc <= 188, f"coder.py has {loc} lines, expected ≤188"


def test_acceptance_criteria_enforcement_preserved():
    """AC2: Acceptance criteria enforcement logic preserved."""
    # Test that acceptance criteria are properly rendered
    issue = {
        "name": "test-issue",
        "title": "Test Issue",
        "acceptance_criteria": ["AC1: Do thing one", "AC2: Do thing two"],
    }
    prompt = coder_task_prompt(issue, "/tmp/worktree")
    assert "## Issue to Implement" in prompt
    assert "**Acceptance Criteria**:" in prompt
    assert "- [ ] AC1: Do thing one" in prompt
    assert "- [ ] AC2: Do thing two" in prompt


def test_test_execution_requirements_maintained():
    """AC3: Test execution requirements maintained."""
    # Test that testing strategy and guidance are included
    issue = {
        "name": "test-issue",
        "title": "Test Issue",
        "testing_strategy": "Create unit tests for all functions",
        "guidance": {"testing_guidance": "Follow TDD approach"},
    }
    prompt = coder_task_prompt(issue, "/tmp/worktree")
    assert "**Testing Strategy**: Create unit tests for all functions" in prompt
    assert "**Testing Guidance (from sprint planner)**: Follow TDD approach" in prompt
    # Verify task instructions mention tests
    assert "Write or update tests per the Testing Strategy/guidance" in prompt
    assert "Run tests and report results (tests_passed, test_summary)" in prompt


def test_tool_usage_instructions_complete():
    """AC4: Tool usage instructions remain complete."""
    # Verify SYSTEM_PROMPT contains all tool references
    assert "## Tools Available" in SYSTEM_PROMPT
    assert "READ / WRITE / EDIT files" in SYSTEM_PROMPT
    assert "BASH for running commands (tests, builds, git)" in SYSTEM_PROMPT
    assert "GLOB / GREP for searching the codebase" in SYSTEM_PROMPT


def test_git_operation_guidelines_preserved():
    """AC5: Git operation guidelines preserved."""
    # Verify SYSTEM_PROMPT contains git rules
    assert "## Git Rules" in SYSTEM_PROMPT
    assert "You are working in an isolated worktree" in SYSTEM_PROMPT
    assert "Commit your work when implementation is complete" in SYSTEM_PROMPT
    assert "Do NOT push" in SYSTEM_PROMPT
    assert "Do NOT create new branches" in SYSTEM_PROMPT
    assert "Do NOT add any `Co-Authored-By` trailers" in SYSTEM_PROMPT
    # Verify git context is included in task prompt
    issue = {
        "name": "test-issue",
        "title": "Test Issue",
        "integration_branch": "feature/test",
    }
    prompt = coder_task_prompt(issue, "/tmp/worktree")
    assert "## Git Context" in prompt
    assert "Integration branch: `feature/test`" in prompt
    assert "Working in worktree: `/tmp/worktree`" in prompt


def test_functional_reference_issue():
    """Functional test with reference issue - verify behavior equivalence.

    Tests that the compressed prompt maintains all functional behavior by
    exercising the coder_task_prompt function with a realistic issue.
    """
    # Reference issue with all fields populated
    issue = {
        "name": "add-feature-x",
        "title": "Add Feature X",
        "description": "Implement feature X with Y and Z",
        "acceptance_criteria": [
            "AC1: Feature X is implemented",
            "AC2: Tests pass",
            "AC3: Documentation updated",
        ],
        "depends_on": ["issue-1", "issue-2"],
        "provides": ["feature-x-interface", "feature-x-utils"],
        "files_to_create": ["src/feature_x.py"],
        "files_to_modify": ["src/main.py", "README.md"],
        "testing_strategy": "Create unit tests and integration tests",
        "guidance": {
            "testing_guidance": "Focus on edge cases",
            "review_guidance": "Check performance",
        },
        "integration_branch": "feature/sprint-1",
        "failure_notes": ["Previous attempt failed due to X"],
    }

    project_context = {
        "prd_path": "/path/to/prd.md",
        "architecture_path": "/path/to/architecture.md",
        "issues_dir": "/path/to/issues",
    }

    memory_context = {
        "codebase_conventions": {
            "test_framework": "pytest",
            "style": "black",
        },
        "failure_patterns": [
            {
                "pattern": "import-error",
                "issue": "issue-1",
                "description": "Missing imports",
            }
        ],
        "dependency_interfaces": [
            {
                "issue": "issue-1",
                "summary": "Base classes",
                "exports": ["BaseClass", "HelperMixin"],
            }
        ],
        "bug_patterns": [
            {
                "type": "null-pointer",
                "frequency": 3,
                "modules": ["module-a", "module-b"],
            }
        ],
    }

    # Generate prompt
    prompt = coder_task_prompt(
        issue,
        "/tmp/worktree",
        feedback="",
        iteration=1,
        project_context=project_context,
        memory_context=memory_context,
    )

    # Verify all sections are present
    assert "## Issue to Implement" in prompt
    assert "**Name**: add-feature-x" in prompt
    assert "**Title**: Add Feature X" in prompt
    assert "**Acceptance Criteria**:" in prompt
    assert "- [ ] AC1: Feature X is implemented" in prompt
    assert "**Dependencies**: ['issue-1', 'issue-2']" in prompt
    assert "**Provides**: ['feature-x-interface', 'feature-x-utils']" in prompt
    assert "**Files to create**: ['src/feature_x.py']" in prompt
    assert "**Files to modify**: ['src/main.py', 'README.md']" in prompt
    assert "**Testing Strategy**: Create unit tests and integration tests" in prompt
    assert "**Testing Guidance (from sprint planner)**: Focus on edge cases" in prompt

    # Project context
    assert "## Project Context" in prompt
    assert "- PRD: `/path/to/prd.md` (read for full requirements)" in prompt
    assert "- Architecture: `/path/to/architecture.md` (read for design decisions)" in prompt

    # Memory context
    assert "## Codebase Conventions (from prior issues)" in prompt
    assert "**test_framework**: pytest" in prompt
    assert "## Known Failure Patterns (avoid these)" in prompt
    assert "**import-error** (issue-1): Missing imports" in prompt
    assert "## Dependency Interfaces (completed upstream issues)" in prompt
    assert "**issue-1**: Base classes" in prompt
    assert "`BaseClass`" in prompt
    assert "## Common Bug Patterns in This Build" in prompt
    assert "null-pointer (seen 3x in ['module-a', 'module-b'])" in prompt

    # Failure notes
    assert "## Upstream Failure Notes" in prompt
    assert "- Previous attempt failed due to X" in prompt

    # Git context
    assert "## Git Context" in prompt
    assert "Integration branch: `feature/sprint-1`" in prompt
    assert "Working in worktree: `/tmp/worktree`" in prompt

    # Working directory and iteration
    assert "## Working Directory" in prompt
    assert "`/tmp/worktree`" in prompt
    assert "## Iteration: 1" in prompt

    # Task instructions (no feedback)
    assert "## Your Task" in prompt
    assert "1. Explore the codebase to understand patterns and context." in prompt
    assert "2. Implement the solution per the acceptance criteria." in prompt


def test_feedback_iteration_path():
    """Test that feedback iteration path works correctly."""
    issue = {"name": "test-issue", "title": "Test Issue"}
    feedback = "Fix the following:\n- Issue 1\n- Issue 2"

    prompt = coder_task_prompt(
        issue, "/tmp/worktree", feedback=feedback, iteration=2
    )

    # Verify feedback section replaces task section
    assert "## Feedback from Previous Iteration" in prompt
    assert "Address ALL of the following issues from the review:" in prompt
    assert feedback in prompt
    assert "Fix the issues above, then re-commit" in prompt
    assert "## Your Task" not in prompt  # Task section should not appear with feedback
    assert "## Iteration: 2" in prompt
