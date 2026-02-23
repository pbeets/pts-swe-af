"""Unit tests for baseline timing instrumentation in app.py.

Tests verify that timing tags appear in app.note() calls with correct format.
Covers:
- AC1: Build start timestamp captured at app.py line 166
- AC2: Build duration logged after completion with tags=['build', 'metrics', 'duration_s']
- AC3: PM, Architect, Tech Lead, Sprint Planner durations logged with phase tags
- AC4: All timing tags follow format duration:<seconds> for parsing
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_prd_dict() -> dict[str, Any]:
    """Return a minimal valid PRD dict."""
    return {
        "validated_description": "Test goal.",
        "acceptance_criteria": ["AC-1: test"],
        "must_have": ["feature"],
        "nice_to_have": [],
        "out_of_scope": [],
        "assumptions": [],
        "risks": [],
    }


def _make_architecture_dict() -> dict[str, Any]:
    """Return a minimal valid Architecture dict."""
    return {
        "summary": "Test architecture.",
        "components": [
            {
                "name": "test-component",
                "responsibility": "Test",
                "touches_files": ["test.py"],
                "depends_on": [],
            }
        ],
        "interfaces": [],
        "decisions": [],
        "file_changes_overview": "Test changes.",
    }


def _make_review_approved_dict() -> dict[str, Any]:
    """Return a ReviewResult dict with approved=True."""
    return {
        "approved": True,
        "feedback": "Approved.",
        "scope_issues": [],
        "complexity_assessment": "appropriate",
        "summary": "Approved.",
    }


def _make_sprint_planner_result() -> dict[str, Any]:
    """Return a minimal sprint planner result."""
    return {
        "issues": [
            {
                "name": "test-issue",
                "title": "Test Issue",
                "description": "Test",
                "depends_on": [],
                "provides": ["test"],
                "files_to_create": [],
                "files_to_modify": ["test.py"],
                "acceptance_criteria": ["AC-1: test"],
                "testing_strategy": "Test",
                "guidance": {
                    "needs_new_tests": True,
                    "estimated_scope": "small",
                    "touches_interfaces": False,
                    "needs_deeper_qa": False,
                },
            }
        ],
        "rationale": "Test rationale",
    }


def _make_dag_executor_result() -> dict[str, Any]:
    """Return a minimal DAG executor result."""
    return {
        "completed_issues": ["test-issue"],
        "all_issues": ["test-issue"],
        "failed_issues": [],
        "skipped_issues": [],
        "levels": [[{"name": "test-issue"}]],
    }


def _make_verifier_result() -> dict[str, Any]:
    """Return a minimal verifier result."""
    return {
        "passed": True,
        "summary": "Verification passed",
    }


def test_build_timing_instrumentation_present(mock_agent_ai):
    """Test that build duration timing is logged with correct tags (AC2, AC4).

    Verifies:
    - Build duration is logged after completion
    - Tags include 'build', 'metrics', 'duration_s'
    - Duration tag follows format 'duration:<seconds>'
    """
    from swe_af import app as app_module

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock note to capture all calls
        note_calls = []
        original_note = app_module.app.note

        def mock_note(msg, tags=None):
            note_calls.append({"msg": msg, "tags": tags or []})
            return original_note(msg, tags=tags)

        with patch.object(app_module.app, 'note', side_effect=mock_note):
            # Mock all agent calls with valid responses
            mock_agent_ai.side_effect = [
                # plan() returns PlanResult
                {
                    "prd": _make_prd_dict(),
                    "architecture": _make_architecture_dict(),
                    "issues": _make_sprint_planner_result()["issues"],
                    "rationale": "Test",
                    "levels": [[{"name": "test-issue"}]],
                    "file_conflicts": [],
                    "artifacts_dir": os.path.join(tmpdir, ".artifacts"),
                },
                # git_init
                {
                    "success": True,
                    "integration_branch": "feature/test",
                    "original_branch": "main",
                    "initial_commit_sha": "abc123",
                    "mode": "transient",
                },
                # execute DAG
                _make_dag_executor_result(),
                # verifier
                _make_verifier_result(),
                # finalize
                {"success": True, "summary": "Finalized"},
            ]

            # Run build
            result = _run(
                app_module.build(
                    goal="Test goal",
                    repo_path=tmpdir,
                    artifacts_dir=".artifacts",
                )
            )

            # Verify build duration was logged (AC2)
            build_duration_logs = [
                call for call in note_calls
                if "Build duration:" in call["msg"]
            ]
            assert len(build_duration_logs) == 1, "Build duration should be logged once"

            duration_log = build_duration_logs[0]
            tags = duration_log["tags"]

            # AC2: Verify required tags
            assert "build" in tags, "Build duration log must have 'build' tag"
            assert "metrics" in tags, "Build duration log must have 'metrics' tag"
            assert "duration_s" in tags, "Build duration log must have 'duration_s' tag"

            # AC4: Verify duration tag format
            duration_tags = [tag for tag in tags if tag.startswith("duration:")]
            assert len(duration_tags) == 1, "Should have exactly one duration: tag"

            duration_value = duration_tags[0].split(":")[1]
            # Verify it's a valid float
            float(duration_value)  # Will raise if not valid


def test_planning_phase_timing_instrumentation(mock_agent_ai):
    """Test that PM, Architect, Tech Lead, Sprint Planner durations are logged (AC3, AC4).

    Verifies:
    - PM duration logged with tags=['pipeline', 'pm', 'duration_s']
    - Architect duration logged with tags=['pipeline', 'architect', 'duration_s']
    - Tech Lead duration logged with tags=['pipeline', 'tech_lead', 'duration_s']
    - Sprint Planner duration logged with tags=['pipeline', 'sprint_planner', 'duration_s']
    - All duration tags follow format 'duration:<seconds>'
    """
    from swe_af import app as app_module

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock artifacts directory structure
        artifacts_dir = os.path.join(tmpdir, ".artifacts")
        plan_dir = os.path.join(artifacts_dir, "plan")
        os.makedirs(plan_dir, exist_ok=True)

        # Mock note to capture calls
        note_calls = []
        original_note = app_module.app.note

        def mock_note(msg, tags=None):
            note_calls.append({"msg": msg, "tags": tags or []})
            return original_note(msg, tags=tags)

        with patch.object(app_module.app, 'note', side_effect=mock_note):
            # Mock individual agent responses for plan()
            mock_agent_ai.side_effect = [
                _make_prd_dict(),  # PM
                _make_architecture_dict(),  # Architect
                _make_review_approved_dict(),  # Tech Lead
                _make_sprint_planner_result(),  # Sprint Planner
            ]

            # Run plan()
            result = _run(
                app_module.plan(
                    goal="Test goal",
                    repo_path=tmpdir,
                    artifacts_dir=".artifacts",
                )
            )

            # Verify PM duration (AC3)
            pm_logs = [call for call in note_calls if "PM:" in call["msg"] and "duration_s" in call["tags"]]
            assert len(pm_logs) == 1, "PM duration should be logged once"
            pm_tags = pm_logs[0]["tags"]
            assert "pipeline" in pm_tags
            assert "pm" in pm_tags
            assert "duration_s" in pm_tags
            # AC4: Verify duration format
            pm_duration_tags = [tag for tag in pm_tags if tag.startswith("duration:")]
            assert len(pm_duration_tags) == 1
            float(pm_duration_tags[0].split(":")[1])

            # Verify Architect duration (AC3)
            arch_logs = [call for call in note_calls if "Architect:" in call["msg"] and "duration_s" in call["tags"]]
            assert len(arch_logs) == 1, "Architect duration should be logged once"
            arch_tags = arch_logs[0]["tags"]
            assert "pipeline" in arch_tags
            assert "architect" in arch_tags
            assert "duration_s" in arch_tags
            # AC4: Verify duration format
            arch_duration_tags = [tag for tag in arch_tags if tag.startswith("duration:")]
            assert len(arch_duration_tags) == 1
            float(arch_duration_tags[0].split(":")[1])

            # Verify Tech Lead duration (AC3)
            tl_logs = [call for call in note_calls if "Tech Lead:" in call["msg"] and "duration_s" in call["tags"]]
            assert len(tl_logs) == 1, "Tech Lead duration should be logged once"
            tl_tags = tl_logs[0]["tags"]
            assert "pipeline" in tl_tags
            assert "tech_lead" in tl_tags
            assert "duration_s" in tl_tags
            # AC4: Verify duration format
            tl_duration_tags = [tag for tag in tl_tags if tag.startswith("duration:")]
            assert len(tl_duration_tags) == 1
            float(tl_duration_tags[0].split(":")[1])

            # Verify Sprint Planner duration (AC3)
            sp_logs = [call for call in note_calls if "Sprint Planner:" in call["msg"] and "duration_s" in call["tags"]]
            assert len(sp_logs) == 1, "Sprint Planner duration should be logged once"
            sp_tags = sp_logs[0]["tags"]
            assert "pipeline" in sp_tags
            assert "sprint_planner" in sp_tags
            assert "duration_s" in sp_tags
            # AC4: Verify duration format
            sp_duration_tags = [tag for tag in sp_tags if tag.startswith("duration:")]
            assert len(sp_duration_tags) == 1
            float(sp_duration_tags[0].split(":")[1])


def test_timing_tag_format_parseable(mock_agent_ai):
    """Test that duration tags are parseable as duration:<float> (AC4).

    Verifies that all duration: tags can be parsed to extract numeric seconds.
    """
    from swe_af import app as app_module

    with tempfile.TemporaryDirectory() as tmpdir:
        note_calls = []
        original_note = app_module.app.note

        def mock_note(msg, tags=None):
            note_calls.append({"msg": msg, "tags": tags or []})
            return original_note(msg, tags=tags)

        with patch.object(app_module.app, 'note', side_effect=mock_note):
            # Mock agent responses for plan()
            mock_agent_ai.side_effect = [
                _make_prd_dict(),
                _make_architecture_dict(),
                _make_review_approved_dict(),
                _make_sprint_planner_result(),
            ]

            # Run plan()
            _run(
                app_module.plan(
                    goal="Test goal",
                    repo_path=tmpdir,
                    artifacts_dir=".artifacts",
                )
            )

            # Extract all duration: tags
            duration_tags = []
            for call in note_calls:
                if "duration_s" in call["tags"]:
                    for tag in call["tags"]:
                        if tag.startswith("duration:"):
                            duration_tags.append(tag)

            # Verify all duration tags are parseable (AC4)
            assert len(duration_tags) > 0, "Should have at least one duration tag"

            for tag in duration_tags:
                parts = tag.split(":", 1)
                assert len(parts) == 2, f"Duration tag {tag} should have format 'duration:<value>'"
                assert parts[0] == "duration", f"Tag {tag} should start with 'duration:'"

                # Verify the value is a valid float
                try:
                    value = float(parts[1])
                    assert value >= 0, f"Duration {tag} should be non-negative"
                except ValueError:
                    pytest.fail(f"Duration tag {tag} has non-numeric value: {parts[1]}")
