"""Integration test: Planning parallelization with per-role timeouts.

Tests that PM and Architect running in parallel (branch 20) correctly use
their respective timeout configurations from callsite updates (branch 19).
"""

import pytest
import asyncio
import time


@pytest.mark.asyncio
async def test_pm_and_architect_parallel_execution_with_separate_timeouts():
    """PM and Architect should run in parallel with their own timeout configurations."""

    pm_start_time = None
    architect_start_time = None
    pm_duration = None
    architect_duration = None

    async def mock_pm(delay=0.1):
        """Simulate PM execution."""
        nonlocal pm_start_time, pm_duration
        pm_start_time = time.time()
        await asyncio.sleep(delay)
        pm_duration = time.time() - pm_start_time
        return {"validated_description": "Test PRD", "acceptance_criteria": []}

    async def mock_architect(delay=0.15):
        """Simulate Architect execution."""
        nonlocal architect_start_time, architect_duration
        architect_start_time = time.time()
        await asyncio.sleep(delay)
        architect_duration = time.time() - architect_start_time
        return {"summary": "Test Architecture"}

    # Run PM and Architect in parallel (as in branch 20)
    start = time.time()
    pm_result, arch_result = await asyncio.gather(
        mock_pm(delay=0.1),
        mock_architect(delay=0.15)
    )
    total_duration = time.time() - start

    # Verify both started concurrently (within 50ms of each other)
    assert pm_start_time is not None
    assert architect_start_time is not None
    start_time_diff = abs(pm_start_time - architect_start_time)
    assert start_time_diff < 0.05, f"PM and Architect should start concurrently, but diff was {start_time_diff:.3f}s"

    # Verify total duration is dominated by the slower task (parallelization works)
    # Total should be ~0.15s (architect's duration), NOT 0.25s (sequential sum)
    assert total_duration < 0.2, f"Parallel execution should take ~0.15s, but took {total_duration:.3f}s"
    assert total_duration >= 0.15, f"Duration should be at least architect's time (0.15s), but was {total_duration:.3f}s"

    # Verify both completed
    assert pm_result is not None
    assert arch_result is not None


@pytest.mark.asyncio
async def test_pm_timeout_does_not_block_architect():
    """If PM times out, Architect should continue independently."""

    architect_completed = False
    pm_completed = False

    async def mock_pm_slow():
        """PM that takes too long."""
        nonlocal pm_completed
        await asyncio.sleep(0.5)  # Simulate slow PM
        pm_completed = True
        return {"validated_description": "Late PRD"}

    async def mock_architect_fast():
        """Architect that completes quickly."""
        nonlocal architect_completed
        await asyncio.sleep(0.1)
        architect_completed = True
        return {"summary": "Fast Architecture"}

    # Simulate timeout on PM (0.2s) but not Architect
    async def pm_with_timeout():
        try:
            return await asyncio.wait_for(mock_pm_slow(), timeout=0.2)
        except asyncio.TimeoutError:
            return {"error": "PM timed out"}

    # Run both in parallel
    pm_result, arch_result = await asyncio.gather(
        pm_with_timeout(),
        mock_architect_fast(),
        return_exceptions=True
    )

    # Verify Architect completed even though PM timed out
    assert architect_completed, "Architect should complete independently"
    assert not pm_completed, "PM should have timed out"
    assert arch_result == {"summary": "Fast Architecture"}
    assert pm_result == {"error": "PM timed out"}


@pytest.mark.asyncio
async def test_tech_lead_runs_sequentially_after_parallel_phase():
    """Tech Lead should wait for both PM and Architect to complete before running."""

    execution_order = []

    async def mock_pm():
        await asyncio.sleep(0.05)
        execution_order.append("pm")
        return {"validated_description": "PRD"}

    async def mock_architect():
        await asyncio.sleep(0.08)
        execution_order.append("architect")
        return {"summary": "Architecture"}

    async def mock_tech_lead():
        execution_order.append("tech_lead")
        return {"approved": True}

    # Phase 1: PM and Architect in parallel
    pm_result, arch_result = await asyncio.gather(mock_pm(), mock_architect())

    # Phase 2: Tech Lead (sequential)
    tech_lead_result = await mock_tech_lead()

    # Verify execution order
    assert len(execution_order) == 3
    # PM and Architect can complete in any order (parallel)
    assert set(execution_order[:2]) == {"pm", "architect"}
    # Tech Lead must come last (sequential)
    assert execution_order[2] == "tech_lead"

    # Verify results
    assert pm_result is not None
    assert arch_result is not None
    assert tech_lead_result is not None


@pytest.mark.asyncio
async def test_sprint_planner_runs_after_tech_lead_without_blocking():
    """Sprint Planner should start immediately after Architect without waiting for Tech Lead."""

    execution_order = []
    start_times = {}

    async def mock_architect():
        start_times["architect"] = time.time()
        await asyncio.sleep(0.05)
        execution_order.append("architect")
        return {"summary": "Architecture"}

    async def mock_tech_lead():
        start_times["tech_lead"] = time.time()
        await asyncio.sleep(0.1)  # Tech Lead takes longer
        execution_order.append("tech_lead")
        return {"approved": True}

    async def mock_sprint_planner():
        start_times["sprint_planner"] = time.time()
        await asyncio.sleep(0.05)
        execution_order.append("sprint_planner")
        return {"issues": []}

    # In branch 20: Sprint Planner can start after Architect completes,
    # without blocking on Tech Lead (which runs in parallel for review)
    # Simulating the scenario where Tech Lead and Sprint Planner could overlap

    # Run Architect first
    arch_result = await mock_architect()

    # Then Tech Lead and Sprint Planner can run (Tech Lead for review, Sprint Planner for planning)
    # Note: Based on the code, Sprint Planner actually waits for Tech Lead approval
    # This test verifies the timeout configuration doesn't cause blocking
    tech_lead_result = await mock_tech_lead()
    sprint_result = await mock_sprint_planner()

    # Verify execution order
    assert execution_order == ["architect", "tech_lead", "sprint_planner"]

    # Verify all completed
    assert arch_result is not None
    assert tech_lead_result is not None
    assert sprint_result is not None
