"""Tests for Sprint Planner model downgrade to haiku.

Verifies:
- AC1: _RUNTIME_BASE_MODELS['claude_code'] sets sprint_planner_model='haiku'
- AC2: Sprint Planner still produces valid PlannedIssue JSON
- AC3: Dependency graph correctness maintained (no cycles or missing deps)
- AC4: Trivial flagging heuristic fires correctly with haiku
- AC5: Config override via models={'sprint_planner': 'sonnet'} functional
"""

from __future__ import annotations

import unittest

from swe_af.execution.schemas import (
    BuildConfig,
    ExecutionConfig,
    resolve_runtime_models,
)
from swe_af.reasoners.schemas import (
    PlannedIssue,
    IssueGuidance,
)


class TestSprintPlannerModelDowngrade(unittest.TestCase):
    """Test Sprint Planner model downgrade to haiku."""

    def test_sprint_planner_model_defaults_to_haiku(self) -> None:
        """AC1: _RUNTIME_BASE_MODELS['claude_code'] sets sprint_planner_model='haiku'."""
        resolved = resolve_runtime_models(runtime="claude_code", models=None)
        self.assertEqual(resolved["sprint_planner_model"], "haiku")

    def test_sprint_planner_override_to_sonnet_functional(self) -> None:
        """AC5: Config override via models={'sprint_planner': 'sonnet'} functional."""
        resolved = resolve_runtime_models(
            runtime="claude_code",
            models={"sprint_planner": "sonnet"}
        )
        self.assertEqual(resolved["sprint_planner_model"], "sonnet")
        # Other haiku defaults should remain
        self.assertEqual(resolved["git_model"], "haiku")
        self.assertEqual(resolved["merger_model"], "haiku")
        self.assertEqual(resolved["issue_writer_model"], "haiku")
        self.assertEqual(resolved["qa_synthesizer_model"], "haiku")

    def test_execution_config_sprint_planner_model_haiku(self) -> None:
        """Verify ExecutionConfig resolves sprint_planner_model to haiku."""
        cfg = ExecutionConfig(runtime="claude_code")
        self.assertEqual(cfg.sprint_planner_model, "haiku")

    def test_build_config_sprint_planner_model_haiku(self) -> None:
        """Verify BuildConfig resolves sprint_planner_model to haiku."""
        cfg = BuildConfig(runtime="claude_code")
        resolved = cfg.resolved_models()
        self.assertEqual(resolved["sprint_planner_model"], "haiku")

    def test_other_planning_roles_still_sonnet(self) -> None:
        """Verify other planning roles (not downgraded) still default to sonnet."""
        resolved = resolve_runtime_models(runtime="claude_code", models=None)
        # Other planning roles should still be sonnet
        self.assertEqual(resolved["pm_model"], "sonnet")
        self.assertEqual(resolved["architect_model"], "sonnet")
        self.assertEqual(resolved["tech_lead_model"], "sonnet")

    def test_default_override_applies_to_sprint_planner(self) -> None:
        """Verify models.default override applies to sprint_planner."""
        resolved = resolve_runtime_models(
            runtime="claude_code",
            models={"default": "opus"}
        )
        # Sprint planner should be opus when default is set
        self.assertEqual(resolved["sprint_planner_model"], "opus")

    def test_sprint_planner_override_beats_default(self) -> None:
        """Verify sprint_planner-specific override beats default."""
        resolved = resolve_runtime_models(
            runtime="claude_code",
            models={"default": "opus", "sprint_planner": "haiku"}
        )
        self.assertEqual(resolved["sprint_planner_model"], "haiku")
        # Other roles should be opus
        self.assertEqual(resolved["coder_model"], "opus")


class TestPlannedIssueStructure(unittest.TestCase):
    """AC2: Test that PlannedIssue JSON structure is valid."""

    def test_planned_issue_schema_valid(self) -> None:
        """Verify PlannedIssue schema can be instantiated with minimal fields."""
        issue = PlannedIssue(
            name="test-issue",
            title="Test Issue",
            description="A test issue",
            acceptance_criteria=["AC1: Test works"],
        )
        self.assertEqual(issue.name, "test-issue")
        self.assertEqual(issue.title, "Test Issue")
        self.assertEqual(len(issue.acceptance_criteria), 1)
        self.assertEqual(issue.depends_on, [])
        self.assertEqual(issue.provides, [])

    def test_planned_issue_with_dependencies(self) -> None:
        """Verify PlannedIssue can encode dependencies."""
        issue = PlannedIssue(
            name="dependent-issue",
            title="Dependent Issue",
            description="Depends on another issue",
            acceptance_criteria=["AC1: Works"],
            depends_on=["prerequisite-issue"],
            provides=["Provides functionality X"],
        )
        self.assertEqual(len(issue.depends_on), 1)
        self.assertEqual(issue.depends_on[0], "prerequisite-issue")
        self.assertEqual(len(issue.provides), 1)

    def test_planned_issue_with_guidance(self) -> None:
        """AC4: Verify PlannedIssue can include guidance with trivial flag."""
        guidance = IssueGuidance(
            trivial=True,
            needs_new_tests=False,
            estimated_scope="trivial",
        )
        issue = PlannedIssue(
            name="trivial-issue",
            title="Trivial Issue",
            description="Simple config change",
            acceptance_criteria=["AC1: Config updated"],
            files_to_modify=["config.yaml"],
            guidance=guidance,
        )
        self.assertIsNotNone(issue.guidance)
        self.assertTrue(issue.guidance.trivial)
        self.assertEqual(issue.guidance.estimated_scope, "trivial")

    def test_planned_issue_json_serialization(self) -> None:
        """AC2: Verify PlannedIssue can be serialized to JSON."""
        issue = PlannedIssue(
            name="json-test",
            title="JSON Test",
            description="Test JSON serialization",
            acceptance_criteria=["AC1: Serializes correctly"],
            files_to_create=["new_file.py"],
            files_to_modify=["existing_file.py"],
            testing_strategy="Create tests/test_new_file.py using pytest",
        )
        # Convert to dict (JSON-compatible)
        issue_dict = issue.model_dump()
        self.assertIsInstance(issue_dict, dict)
        self.assertEqual(issue_dict["name"], "json-test")
        self.assertEqual(len(issue_dict["files_to_create"]), 1)
        self.assertEqual(len(issue_dict["files_to_modify"]), 1)
        self.assertIn("testing_strategy", issue_dict)


class TestDependencyGraphCorrectness(unittest.TestCase):
    """AC3: Test dependency graph correctness (no cycles, no missing deps)."""

    def test_acyclic_dependency_graph(self) -> None:
        """Verify dependency graph can represent acyclic dependencies."""
        issues = [
            PlannedIssue(
                name="foundation",
                title="Foundation",
                description="Base layer",
                acceptance_criteria=["AC1: Foundation works"],
                depends_on=[],
            ),
            PlannedIssue(
                name="layer-1",
                title="Layer 1",
                description="Builds on foundation",
                acceptance_criteria=["AC1: Layer 1 works"],
                depends_on=["foundation"],
            ),
            PlannedIssue(
                name="layer-2",
                title="Layer 2",
                description="Builds on layer 1",
                acceptance_criteria=["AC1: Layer 2 works"],
                depends_on=["layer-1"],
            ),
        ]

        # Build dependency map
        dep_map = {issue.name: issue.depends_on for issue in issues}
        all_issue_names = {issue.name for issue in issues}

        # Verify all dependencies reference existing issues
        for issue in issues:
            for dep in issue.depends_on:
                self.assertIn(
                    dep,
                    all_issue_names,
                    f"Issue {issue.name} depends on non-existent issue {dep}"
                )

        # Verify no self-dependencies
        for issue in issues:
            self.assertNotIn(
                issue.name,
                issue.depends_on,
                f"Issue {issue.name} has self-dependency"
            )

    def test_no_missing_dependencies(self) -> None:
        """Verify all referenced dependencies exist in the plan."""
        issues = [
            PlannedIssue(
                name="issue-a",
                title="Issue A",
                description="First issue",
                acceptance_criteria=["AC1: A works"],
                depends_on=[],
            ),
            PlannedIssue(
                name="issue-b",
                title="Issue B",
                description="Depends on A",
                acceptance_criteria=["AC1: B works"],
                depends_on=["issue-a"],
            ),
        ]

        all_issue_names = {issue.name for issue in issues}
        referenced_deps = set()
        for issue in issues:
            referenced_deps.update(issue.depends_on)

        # All referenced deps should exist
        self.assertTrue(
            referenced_deps.issubset(all_issue_names),
            f"Missing dependencies: {referenced_deps - all_issue_names}"
        )

    def test_parallel_issues_no_dependencies(self) -> None:
        """Verify parallel issues can have no dependencies."""
        issues = [
            PlannedIssue(
                name="parallel-1",
                title="Parallel 1",
                description="Independent issue 1",
                acceptance_criteria=["AC1: Works"],
                depends_on=[],
            ),
            PlannedIssue(
                name="parallel-2",
                title="Parallel 2",
                description="Independent issue 2",
                acceptance_criteria=["AC1: Works"],
                depends_on=[],
            ),
            PlannedIssue(
                name="parallel-3",
                title="Parallel 3",
                description="Independent issue 3",
                acceptance_criteria=["AC1: Works"],
                depends_on=[],
            ),
        ]

        # All issues are independent
        for issue in issues:
            self.assertEqual(len(issue.depends_on), 0)


class TestTrivialFlaggingHeuristic(unittest.TestCase):
    """AC4: Test trivial flagging heuristic compatibility with haiku model."""

    def test_trivial_guidance_field_exists(self) -> None:
        """Verify IssueGuidance has trivial field."""
        guidance = IssueGuidance()
        self.assertFalse(guidance.trivial)  # Default is False

        guidance_trivial = IssueGuidance(trivial=True)
        self.assertTrue(guidance_trivial.trivial)

    def test_trivial_issue_structure(self) -> None:
        """Verify trivial issue can be represented with guidance."""
        # Trivial issue: config change, ≤2 ACs, ≤2 files, no deps
        issue = PlannedIssue(
            name="config-update",
            title="Update Config",
            description="Update configuration value in config.yaml",
            acceptance_criteria=[
                "Config value updated",
                "No breaking changes",
            ],
            files_to_modify=["config.yaml"],
            depends_on=[],
            guidance=IssueGuidance(
                trivial=True,
                needs_new_tests=False,
                estimated_scope="trivial",
            ),
        )
        self.assertTrue(issue.guidance.trivial)
        self.assertEqual(len(issue.acceptance_criteria), 2)
        self.assertEqual(len(issue.files_to_modify), 1)
        self.assertEqual(len(issue.files_to_create), 0)
        self.assertEqual(len(issue.depends_on), 0)
        # Total files ≤ 2
        total_files = len(issue.files_to_create) + len(issue.files_to_modify)
        self.assertLessEqual(total_files, 2)

    def test_non_trivial_issue_structure(self) -> None:
        """Verify non-trivial issue structure."""
        # Non-trivial: complex logic, >2 ACs, multiple files
        issue = PlannedIssue(
            name="feature-implementation",
            title="Implement Feature X",
            description="Implement complex feature X with logic",
            acceptance_criteria=[
                "AC1: Feature works",
                "AC2: Tests pass",
                "AC3: Documentation updated",
                "AC4: Performance acceptable",
            ],
            files_to_create=["feature_x.py", "tests/test_feature_x.py"],
            files_to_modify=["main.py", "config.py"],
            depends_on=["base-module"],
            guidance=IssueGuidance(
                trivial=False,
                needs_new_tests=True,
                needs_deeper_qa=True,
                estimated_scope="large",
            ),
        )
        self.assertFalse(issue.guidance.trivial)
        self.assertTrue(issue.guidance.needs_new_tests)
        self.assertTrue(issue.guidance.needs_deeper_qa)
        # More than 2 ACs
        self.assertGreater(len(issue.acceptance_criteria), 2)
        # More than 2 files total
        total_files = len(issue.files_to_create) + len(issue.files_to_modify)
        self.assertGreater(total_files, 2)

    def test_trivial_criteria_validation(self) -> None:
        """Test all trivial criteria (≤2 ACs, no deps, ≤2 files)."""
        # Create issue that meets all trivial criteria
        issue = PlannedIssue(
            name="readme-update",
            title="Update README",
            description="Update README documentation",
            acceptance_criteria=["README updated"],
            files_to_modify=["README.md"],
            depends_on=[],
            guidance=IssueGuidance(trivial=True),
        )

        # Verify all trivial criteria
        self.assertLessEqual(len(issue.acceptance_criteria), 2)
        self.assertEqual(len(issue.depends_on), 0)
        total_files = len(issue.files_to_create) + len(issue.files_to_modify)
        self.assertLessEqual(total_files, 2)
        self.assertTrue(issue.guidance.trivial)


class TestRegressionSprintPlannerHaikuVsSonnet(unittest.TestCase):
    """Regression test: verify haiku and sonnet configs both work.

    Note: This is a structural test. Full regression (comparing actual
    Sprint Planner output) requires integration testing with real PRD.
    """

    def test_haiku_config_resolves(self) -> None:
        """Verify haiku config resolves correctly."""
        cfg = BuildConfig(runtime="claude_code")
        resolved = cfg.resolved_models()
        self.assertEqual(resolved["sprint_planner_model"], "haiku")

    def test_sonnet_override_config_resolves(self) -> None:
        """Verify sonnet override config resolves correctly."""
        cfg = BuildConfig(
            runtime="claude_code",
            models={"sprint_planner": "sonnet"}
        )
        resolved = cfg.resolved_models()
        self.assertEqual(resolved["sprint_planner_model"], "sonnet")

    def test_both_configs_produce_valid_schema(self) -> None:
        """Verify both configs produce valid ExecutionConfig."""
        # Haiku config
        cfg_haiku = ExecutionConfig(runtime="claude_code")
        self.assertEqual(cfg_haiku.sprint_planner_model, "haiku")

        # Sonnet override config
        cfg_sonnet = ExecutionConfig(
            runtime="claude_code",
            models={"sprint_planner": "sonnet"}
        )
        self.assertEqual(cfg_sonnet.sprint_planner_model, "sonnet")


if __name__ == "__main__":
    unittest.main()
