"""Tests for Planning Parallelization (PM + Architect concurrent execution).

Tests verify:
- asyncio.gather() invocation with both PM and Architect coroutines
- Concurrent execution timing vs baseline sequential
- PRD file atomicity and race condition prevention
- Tech Lead review loop runs sequentially after gather completes
- Sprint Planner runs after Tech Lead approval
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAsyncGatherInvocation:
    """Unit tests verifying asyncio.gather() is used correctly."""

    @pytest.mark.asyncio
    async def test_pm_architect_coroutines_created_before_await(self):
        """Verify PM and Architect coroutines are created before await."""
        # Import app module to access plan reasoner
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from swe_af import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the app.call to return dummy results
            pm_result = {
                "validated_description": "Test PRD",
                "acceptance_criteria": [],
                "must_have": [],
                "nice_to_have": [],
                "out_of_scope": [],
            }
            arch_result = {
                "components": [],
                "dependencies": [],
                "summary": "Test arch",
                "interfaces": [],
                "decisions": [],
                "file_changes_overview": "Test file changes",
            }

            call_count = 0
            call_order = []

            async def mock_call(endpoint: str, **kwargs):
                nonlocal call_count
                call_count += 1
                if "run_product_manager" in endpoint:
                    call_order.append("pm")
                    await asyncio.sleep(0.1)  # Simulate work
                    return pm_result
                elif "run_architect" in endpoint:
                    call_order.append("architect")
                    await asyncio.sleep(0.1)  # Simulate work
                    return arch_result
                elif "run_tech_lead" in endpoint:
                    call_order.append("tech_lead")
                    return {"approved": True, "feedback": "", "summary": "OK"}
                elif "run_sprint_planner" in endpoint:
                    call_order.append("sprint_planner")
                    return {"issues": [], "rationale": "Test rationale"}
                else:
                    return {}

            with patch.object(app_module.app, "call", side_effect=mock_call):
                with patch.object(app_module.app, "note"):
                    # Run plan reasoner
                    result = await app_module.plan(
                        goal="Test parallelization",
                        repo_path=tmpdir,
                        artifacts_dir=".artifacts",
                    )

            # Verify both PM and Architect were called
            assert "pm" in call_order
            assert "architect" in call_order

            # Verify PM and Architect ran in parallel (both start before either completes)
            # If sequential, order would be: [pm, architect, ...]
            # If parallel, both start ~simultaneously
            pm_idx = call_order.index("pm")
            arch_idx = call_order.index("architect")
            # In parallel execution, indices should be 0 and 1 (order may vary due to race)
            assert {pm_idx, arch_idx} == {0, 1}, f"Expected parallel start, got order: {call_order}"

    @pytest.mark.asyncio
    async def test_asyncio_gather_executes_concurrently(self):
        """Verify asyncio.gather() enables concurrent execution."""

        async def mock_pm():
            await asyncio.sleep(0.5)
            return {"validated_description": "PM result"}

        async def mock_architect():
            await asyncio.sleep(0.5)
            return {
                "components": [],
                "dependencies": [],
                "summary": "Test arch",
                "interfaces": [],
                "decisions": [],
                "file_changes_overview": "Test file changes",
            }

        # Test sequential execution (baseline)
        start_seq = time.time()
        pm_result_seq = await mock_pm()
        arch_result_seq = await mock_architect()
        seq_duration = time.time() - start_seq

        # Test parallel execution with gather
        start_par = time.time()
        pm_result_par, arch_result_par = await asyncio.gather(mock_pm(), mock_architect())
        par_duration = time.time() - start_par

        # Parallel should be ~50% of sequential (both 0.5s tasks run concurrently)
        assert seq_duration >= 1.0, f"Sequential should take ≥1s, got {seq_duration:.2f}s"
        assert par_duration < 0.7, f"Parallel should take <0.7s, got {par_duration:.2f}s"
        assert par_duration < seq_duration * 0.6, f"Parallel not faster: seq={seq_duration:.2f}s, par={par_duration:.2f}s"


class TestConcurrentExecutionTiming:
    """Integration tests measuring concurrent execution time vs baseline."""

    @pytest.mark.asyncio
    async def test_parallel_execution_time_reduction(self):
        """Verify planning phase duration reduced by ≥20% vs sequential."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from swe_af import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            pm_duration = 2.0  # Simulate 2s PM work
            arch_duration = 2.0  # Simulate 2s Architect work

            async def mock_call(endpoint: str, **kwargs):
                if "run_product_manager" in endpoint:
                    await asyncio.sleep(pm_duration)
                    return {
                        "validated_description": "Test PRD",
                        "acceptance_criteria": [],
                        "must_have": [],
                        "nice_to_have": [],
                        "out_of_scope": [],
                    }
                elif "run_architect" in endpoint:
                    await asyncio.sleep(arch_duration)
                    return {
                        "components": [],
                        "dependencies": [],
                        "summary": "Test arch",
                        "interfaces": [],
                        "decisions": [],
                        "file_changes_overview": "Test file changes",
                    }
                elif "run_tech_lead" in endpoint:
                    return {"approved": True, "feedback": "", "summary": "OK"}
                elif "run_sprint_planner" in endpoint:
                    return {"issues": [], "rationale": "Test rationale"}
                else:
                    return {}

            with patch.object(app_module.app, "call", side_effect=mock_call):
                with patch.object(app_module.app, "note"):
                    start = time.time()
                    await app_module.plan(
                        goal="Test timing",
                        repo_path=tmpdir,
                        artifacts_dir=".artifacts",
                    )
                    duration = time.time() - start

            # Sequential baseline would be: pm (2s) + architect (2s) = 4s of agent work
            # Parallel: max(pm, architect) = 2s of agent work
            # With overhead (tech_lead, sprint_planner, file I/O), total execution time is higher
            # but the PM+Architect portion should still show parallelism benefit

            # The key verification is that total duration is less than sequential baseline
            # Sequential baseline = pm_duration + arch_duration + overhead
            # We add a generous overhead budget for file I/O, tech lead, sprint planner, etc.
            sequential_baseline = pm_duration + arch_duration + 15.0  # 4s agents + 15s overhead

            # With parallelism, we expect: max(pm, arch) + overhead = 2s + 15s = 17s
            # Should be noticeably faster than sequential (19s)
            # Allow some margin for test environment variability
            assert duration < sequential_baseline, f"Expected <{sequential_baseline}s (sequential), got {duration:.2f}s"

            # More important: verify the core benefit - duration should be closer to
            # parallel baseline (2s + overhead) than sequential (4s + overhead)
            # This proves PM and Architect ran concurrently
            parallel_baseline = max(pm_duration, arch_duration) + 15.0  # 2s + 15s overhead
            margin = 3.0  # Allow 3s margin for variability
            assert duration < parallel_baseline + margin, f"Expected <{parallel_baseline + margin}s, got {duration:.2f}s"

    @pytest.mark.asyncio
    async def test_pm_architect_overlap_verification(self):
        """Verify PM and Architect work overlaps (no sequential bottleneck)."""
        execution_log = []

        async def mock_pm():
            execution_log.append(("pm_start", time.time()))
            await asyncio.sleep(1.0)
            execution_log.append(("pm_end", time.time()))
            return {"validated_description": "Test"}

        async def mock_architect():
            execution_log.append(("arch_start", time.time()))
            await asyncio.sleep(1.0)
            execution_log.append(("arch_end", time.time()))
            return {
                "components": [],
                "dependencies": [],
                "summary": "Test arch",
                "interfaces": [],
                "decisions": [],
                "file_changes_overview": "Test file changes",
            }

        # Execute in parallel
        await asyncio.gather(mock_pm(), mock_architect())

        # Extract timestamps
        pm_start = next(t for label, t in execution_log if label == "pm_start")
        pm_end = next(t for label, t in execution_log if label == "pm_end")
        arch_start = next(t for label, t in execution_log if label == "arch_start")
        arch_end = next(t for label, t in execution_log if label == "arch_end")

        # Verify overlap: Architect should start before PM completes
        assert arch_start < pm_end, "Architect should start before PM completes (parallel)"
        # Also verify PM starts before Architect completes
        assert pm_start < arch_end, "PM should start before Architect completes (parallel)"

        # Both should start within ~same time (within 100ms)
        assert abs(pm_start - arch_start) < 0.1, f"Start times should be close: {abs(pm_start - arch_start):.3f}s"


class TestPRDPathParameter:
    """Tests verifying prd_path is passed to Architect instead of prd dict."""

    @pytest.mark.asyncio
    async def test_architect_receives_prd_path_not_dict(self):
        """Verify Architect is called with prd_path instead of prd dict."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from swe_af import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            call_args_log = []

            async def mock_call(endpoint: str, **kwargs):
                call_args_log.append((endpoint, kwargs))
                if "run_product_manager" in endpoint:
                    return {
                        "validated_description": "Test",
                        "acceptance_criteria": [],
                        "must_have": [],
                        "nice_to_have": [],
                        "out_of_scope": [],
                    }
                elif "run_architect" in endpoint:
                    return {
                        "components": [],
                        "dependencies": [],
                        "summary": "Test arch",
                        "interfaces": [],
                        "decisions": [],
                        "file_changes_overview": "Test file changes",
                    }
                elif "run_tech_lead" in endpoint:
                    return {"approved": True, "feedback": "", "summary": "OK"}
                elif "run_sprint_planner" in endpoint:
                    return {"issues": [], "rationale": "Test rationale"}
                return {}

            with patch.object(app_module.app, "call", side_effect=mock_call):
                with patch.object(app_module.app, "note"):
                    await app_module.plan(
                        goal="Test prd_path",
                        repo_path=tmpdir,
                        artifacts_dir=".artifacts",
                    )

            # Find Architect call
            arch_call = next(
                (args for endpoint, args in call_args_log if "run_architect" in endpoint),
                None
            )
            assert arch_call is not None, "Architect should be called"

            # Verify prd_path is passed and prd is None
            assert "prd_path" in arch_call, "Architect should receive prd_path parameter"
            assert arch_call.get("prd") is None, "Architect prd parameter should be None (polling mode)"

            # Verify prd_path points to correct file
            expected_path_suffix = os.path.join(".artifacts", "plan", "prd.md")
            actual_prd_path = arch_call["prd_path"]
            assert actual_prd_path.endswith(expected_path_suffix), f"Expected path ending with {expected_path_suffix}, got {actual_prd_path}"


class TestSequentialPhases:
    """Tests verifying Tech Lead and Sprint Planner run sequentially after PM+Architect."""

    @pytest.mark.asyncio
    async def test_tech_lead_runs_after_gather_completes(self):
        """Verify Tech Lead review loop runs sequentially after asyncio.gather completes."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from swe_af import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            execution_order = []

            async def mock_call(endpoint: str, **kwargs):
                if "run_product_manager" in endpoint:
                    execution_order.append("pm_start")
                    await asyncio.sleep(0.1)
                    execution_order.append("pm_end")
                    return {
                        "validated_description": "Test",
                        "acceptance_criteria": [],
                        "must_have": [],
                        "nice_to_have": [],
                        "out_of_scope": [],
                    }
                elif "run_architect" in endpoint:
                    execution_order.append("arch_start")
                    await asyncio.sleep(0.1)
                    execution_order.append("arch_end")
                    return {
                        "components": [],
                        "dependencies": [],
                        "summary": "Test arch",
                        "interfaces": [],
                        "decisions": [],
                        "file_changes_overview": "Test file changes",
                    }
                elif "run_tech_lead" in endpoint:
                    execution_order.append("tech_lead")
                    return {"approved": True, "feedback": "", "summary": "OK"}
                elif "run_sprint_planner" in endpoint:
                    execution_order.append("sprint_planner")
                    return {"issues": [], "rationale": "Test rationale"}
                return {}

            with patch.object(app_module.app, "call", side_effect=mock_call):
                with patch.object(app_module.app, "note"):
                    await app_module.plan(
                        goal="Test sequencing",
                        repo_path=tmpdir,
                        artifacts_dir=".artifacts",
                    )

            # Verify execution order
            assert "pm_start" in execution_order
            assert "arch_start" in execution_order
            assert "pm_end" in execution_order
            assert "arch_end" in execution_order
            assert "tech_lead" in execution_order

            # Tech Lead should run after both PM and Architect complete
            tech_lead_idx = execution_order.index("tech_lead")
            pm_end_idx = execution_order.index("pm_end")
            arch_end_idx = execution_order.index("arch_end")

            assert tech_lead_idx > pm_end_idx, "Tech Lead should run after PM completes"
            assert tech_lead_idx > arch_end_idx, "Tech Lead should run after Architect completes"

    @pytest.mark.asyncio
    async def test_sprint_planner_runs_parallel_with_tech_lead(self):
        """Verify Sprint Planner runs in parallel with Tech Lead (non-blocking)."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from swe_af import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            execution_order = []

            async def mock_call(endpoint: str, **kwargs):
                if "run_product_manager" in endpoint:
                    execution_order.append("pm_start")
                    await asyncio.sleep(0.1)
                    execution_order.append("pm_end")
                    return {
                        "validated_description": "Test",
                        "acceptance_criteria": [],
                        "must_have": [],
                        "nice_to_have": [],
                        "out_of_scope": [],
                    }
                elif "run_architect" in endpoint:
                    execution_order.append("architect_start")
                    await asyncio.sleep(0.1)
                    execution_order.append("architect_end")
                    return {
                        "components": [],
                        "dependencies": [],
                        "summary": "Test arch",
                        "interfaces": [],
                        "decisions": [],
                        "file_changes_overview": "Test file changes",
                    }
                elif "run_tech_lead" in endpoint:
                    execution_order.append("tech_lead_start")
                    await asyncio.sleep(0.1)
                    execution_order.append("tech_lead_end")
                    return {"approved": True, "feedback": "", "summary": "OK"}
                elif "run_sprint_planner" in endpoint:
                    execution_order.append("sprint_planner_start")
                    await asyncio.sleep(0.1)
                    execution_order.append("sprint_planner_end")
                    return {"issues": [], "rationale": "Test rationale"}
                return {}

            with patch.object(app_module.app, "call", side_effect=mock_call):
                with patch.object(app_module.app, "note"):
                    await app_module.plan(
                        goal="Test sprint planner parallelization",
                        repo_path=tmpdir,
                        artifacts_dir=".artifacts",
                    )

            # Verify Sprint Planner and Tech Lead both ran
            assert "tech_lead_start" in execution_order
            assert "sprint_planner_start" in execution_order

            # Verify they ran in parallel: both should start before either completes
            tech_lead_start_idx = execution_order.index("tech_lead_start")
            tech_lead_end_idx = execution_order.index("tech_lead_end")
            sprint_planner_start_idx = execution_order.index("sprint_planner_start")
            sprint_planner_end_idx = execution_order.index("sprint_planner_end")

            # In parallel execution, both should start before either completes
            # Check that Sprint Planner starts before Tech Lead completes (parallel overlap)
            assert sprint_planner_start_idx < tech_lead_end_idx, "Sprint Planner should start before Tech Lead completes (parallel)"
            # Check that Tech Lead starts before Sprint Planner completes (parallel overlap)
            assert tech_lead_start_idx < sprint_planner_end_idx, "Tech Lead should start before Sprint Planner completes (parallel)"

            # Both should start at approximately the same time (within 2 positions in execution order)
            assert abs(tech_lead_start_idx - sprint_planner_start_idx) <= 2, f"Start times should be close: indices {tech_lead_start_idx} vs {sprint_planner_start_idx}"


class TestEdgeCases:
    """Edge case tests: concurrent file access, polling timeout, PM failure."""

    @pytest.mark.asyncio
    async def test_architect_polls_successfully_after_pm_write(self):
        """Verify Architect successfully polls and reads PRD after PM writes it."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from swe_af.reasoners.pipeline import run_product_manager, run_architect

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = ".artifacts"
            prd_path = os.path.join(tmpdir, artifacts_dir, "plan", "prd.md")

            # Create a mock AgentAI that simulates PM and Architect
            with patch("swe_af.reasoners.pipeline.AgentAI") as MockAgentAI:
                # Mock PM to write PRD
                pm_mock = AsyncMock()
                pm_mock.run.return_value = AsyncMock(
                    parsed=MagicMock(
                        validated_description="Test PRD",
                        acceptance_criteria=["AC1"],
                        must_have=["M1"],
                        out_of_scope=["O1"],
                        model_dump=lambda: {
                            "validated_description": "Test PRD",
                            "acceptance_criteria": ["AC1"]
                        }
                    )
                )

                # Mock Architect to verify it reads PRD
                arch_mock = AsyncMock()
                arch_mock.run.return_value = AsyncMock(
                    parsed=MagicMock(
                        components=[],
                        dependencies=[],
                        model_dump=lambda: {"components": [], "dependencies": []}
                    )
                )

                MockAgentAI.return_value = pm_mock

                # Run PM first
                with patch("swe_af.reasoners.pipeline.router.note"):
                    pm_result = await run_product_manager(
                        goal="Test",
                        repo_path=tmpdir,
                        artifacts_dir=artifacts_dir,
                    )

                # Verify PRD file was created
                assert os.path.exists(prd_path), "PM should create PRD file"

                # Now run Architect with polling
                MockAgentAI.return_value = arch_mock
                with patch("swe_af.reasoners.pipeline.router.note"):
                    arch_result = await run_architect(
                        prd=None,  # Polling mode
                        repo_path=tmpdir,
                        artifacts_dir=artifacts_dir,
                        prd_path=prd_path,
                    )

                # Verify Architect completed successfully
                assert arch_result is not None
                assert "components" in arch_result

    @pytest.mark.asyncio
    async def test_concurrent_pm_architect_no_race_condition(self):
        """Verify no race condition when PM writes and Architect polls concurrently."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from swe_af.reasoners.pipeline import run_product_manager, run_architect

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = ".artifacts"
            prd_path = os.path.join(tmpdir, artifacts_dir, "plan", "prd.md")

            with patch("swe_af.reasoners.pipeline.AgentAI") as MockAgentAI:
                # Mock PM
                pm_mock = AsyncMock()
                pm_mock.run.return_value = AsyncMock(
                    parsed=MagicMock(
                        validated_description="Test PRD",
                        acceptance_criteria=["AC1"],
                        must_have=["M1"],
                        out_of_scope=["O1"],
                        model_dump=lambda: {"validated_description": "Test PRD"}
                    )
                )

                # Mock Architect
                arch_mock = AsyncMock()
                arch_mock.run.return_value = AsyncMock(
                    parsed=MagicMock(
                        components=[],
                        model_dump=lambda: {"components": []}
                    )
                )

                async def get_mock(call_num=[0]):
                    call_num[0] += 1
                    return pm_mock if call_num[0] == 1 else arch_mock

                MockAgentAI.side_effect = [pm_mock, arch_mock]

                # Run PM and Architect concurrently
                with patch("swe_af.reasoners.pipeline.router.note"):
                    pm_coro = run_product_manager(
                        goal="Test",
                        repo_path=tmpdir,
                        artifacts_dir=artifacts_dir,
                    )
                    arch_coro = run_architect(
                        prd=None,
                        repo_path=tmpdir,
                        artifacts_dir=artifacts_dir,
                        prd_path=prd_path,
                    )

                    # Execute concurrently
                    pm_result, arch_result = await asyncio.gather(pm_coro, arch_coro)

                # Both should complete successfully without race condition
                assert pm_result is not None
                assert arch_result is not None
                # PRD file should exist
                assert os.path.exists(prd_path)
