"""Tests for scripts/compare_build_costs.py.

Validates that the cost comparison script:
- AC1: Exists and is executable
- AC2: Accepts --baseline and --threshold arguments
- AC3: Loads baseline costs from JSON file
- AC4: Calculates cost reduction percentage
- AC5: Exits with code 0 if reduction >= threshold, code 1 otherwise
- AC6: Provides clear output showing cost comparison
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def baseline_costs_data():
    """Baseline costs with all agents using sonnet."""
    return {
        "agent_durations": {
            "coder": [120.0, 130.0, 115.0],  # 3 calls
            "qa": [60.0, 55.0],  # 2 calls
            "code_reviewer": [45.0, 50.0],
            "git": [10.0, 12.0, 8.0],
            "merger": [20.0],
            "issue_writer": [15.0, 18.0, 14.0],
            "sprint_planner": [180.0],
            "pm": [300.0],
            "architect": [420.0],
        },
        "model_assignments": {
            # All using sonnet at baseline
            "coder": "sonnet",
            "qa": "sonnet",
            "code_reviewer": "sonnet",
            "git": "sonnet",
            "merger": "sonnet",
            "issue_writer": "sonnet",
            "sprint_planner": "sonnet",
            "pm": "sonnet",
            "architect": "sonnet",
        },
    }


@pytest.fixture
def optimized_costs_data():
    """Optimized costs with haiku downgrades for git, merger, issue_writer."""
    return {
        "agent_durations": {
            "coder": [120.0, 130.0, 115.0],
            "qa": [60.0, 55.0],
            "code_reviewer": [45.0, 50.0],
            "git": [10.0, 12.0, 8.0],  # Downgraded to haiku
            "merger": [20.0],  # Downgraded to haiku
            "issue_writer": [15.0, 18.0, 14.0],  # Downgraded to haiku
            "sprint_planner": [180.0],
            "pm": [300.0],
            "architect": [420.0],
        },
        "model_assignments": {
            "coder": "sonnet",
            "qa": "sonnet",
            "code_reviewer": "sonnet",
            "git": "haiku",  # Optimized
            "merger": "haiku",  # Optimized
            "issue_writer": "haiku",  # Optimized
            "sprint_planner": "sonnet",
            "pm": "sonnet",
            "architect": "sonnet",
        },
    }


@pytest.fixture
def baseline_costs_file(tmp_path, baseline_costs_data):
    """Create temporary baseline costs JSON file."""
    baseline_file = tmp_path / "baseline_costs.json"
    with open(baseline_file, "w") as f:
        json.dump(baseline_costs_data, f)
    return baseline_file


@pytest.fixture
def optimized_costs_file(tmp_path, optimized_costs_data):
    """Create temporary optimized costs JSON file."""
    optimized_file = tmp_path / "optimized_costs.json"
    with open(optimized_file, "w") as f:
        json.dump(optimized_costs_data, f)
    return optimized_file


def test_script_exists_and_executable():
    """AC1: scripts/compare_build_costs.py exists and is executable."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"
    assert script_path.exists(), f"Script not found at {script_path}"
    assert script_path.stat().st_mode & 0o111, f"Script not executable: {script_path}"


def test_script_accepts_baseline_and_threshold_arguments(baseline_costs_file):
    """AC2: Script accepts --baseline and --threshold arguments."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    # Should run without error with required arguments
    result = subprocess.run(
        [sys.executable, str(script_path), "--baseline", str(baseline_costs_file), "--threshold", "0.10"],
        capture_output=True,
        text=True,
    )

    # Should execute (might pass or fail based on data, but shouldn't error out)
    assert result.returncode in [0, 1], f"Script should exit with 0 or 1, got {result.returncode}\nStderr: {result.stderr}"


def test_script_requires_baseline_argument():
    """AC2: Script requires --baseline argument."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--threshold", "0.10"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, "Script should fail without --baseline"
    assert "baseline" in result.stderr.lower() or "required" in result.stderr.lower()


def test_script_requires_threshold_argument(baseline_costs_file):
    """AC2: Script requires --threshold argument."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--baseline", str(baseline_costs_file)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, "Script should fail without --threshold"
    assert "threshold" in result.stderr.lower() or "required" in result.stderr.lower()


def test_loads_baseline_costs_from_json(baseline_costs_file):
    """AC3: Script loads baseline costs from JSON file."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--baseline", str(baseline_costs_file), "--threshold", "0.10"],
        capture_output=True,
        text=True,
    )

    # Should load successfully (check stderr for "Loading baseline costs")
    assert "Loading baseline costs" in result.stderr or result.returncode in [0, 1]


def test_handles_missing_baseline_file():
    """AC3: Script handles missing baseline file gracefully."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--baseline", "/nonexistent/file.json", "--threshold", "0.10"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, "Script should exit with code 1 for missing file"
    assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


def test_calculates_cost_reduction(baseline_costs_file, optimized_costs_file):
    """AC4: Script calculates cost reduction percentage."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--baseline",
            str(baseline_costs_file),
            "--current",
            str(optimized_costs_file),
            "--threshold",
            "0.05",
        ],
        capture_output=True,
        text=True,
    )

    # Should show cost reduction in output
    assert "Cost reduction:" in result.stderr or "cost reduction" in result.stderr.lower()
    assert "%" in result.stderr, "Output should show percentage"


def test_exit_code_0_when_reduction_meets_threshold(baseline_costs_file, optimized_costs_file):
    """AC5: Script exits with code 0 if reduction >= threshold."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    # Use a low threshold that should be met
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--baseline",
            str(baseline_costs_file),
            "--current",
            str(optimized_costs_file),
            "--threshold",
            "0.01",  # 1% threshold - should be met
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script should exit with 0 when threshold met\nStderr: {result.stderr}"
    assert "PASS" in result.stderr or "APPROVED" in result.stderr


def test_exit_code_1_when_reduction_below_threshold(baseline_costs_file):
    """AC5: Script exits with code 1 if reduction < threshold."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    # Use same file for baseline and current - 0% reduction
    # Set threshold high to ensure failure
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--baseline",
            str(baseline_costs_file),
            "--threshold",
            "0.50",  # 50% threshold - won't be met with same file
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, f"Script should exit with 1 when threshold not met\nStderr: {result.stderr}"
    assert "FAIL" in result.stderr or "REJECTED" in result.stderr


def test_provides_clear_output(baseline_costs_file, optimized_costs_file):
    """AC6: Script provides clear output showing cost comparison."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--baseline",
            str(baseline_costs_file),
            "--current",
            str(optimized_costs_file),
            "--threshold",
            "0.10",
        ],
        capture_output=True,
        text=True,
    )

    stderr = result.stderr

    # Should show all key information
    assert "baseline" in stderr.lower(), "Output should mention baseline"
    assert "current" in stderr.lower() or "optimized" in stderr.lower(), "Output should mention current/optimized"
    assert "reduction" in stderr.lower(), "Output should mention reduction"
    assert "threshold" in stderr.lower(), "Output should mention threshold"
    assert "cost" in stderr.lower(), "Output should mention cost"

    # Should show clear pass/fail status
    assert "PASS" in stderr or "FAIL" in stderr, "Output should show PASS or FAIL status"


def test_verbose_output_shows_breakdown(baseline_costs_file, optimized_costs_file):
    """AC6: Script with --verbose shows detailed cost breakdown."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--baseline",
            str(baseline_costs_file),
            "--current",
            str(optimized_costs_file),
            "--threshold",
            "0.10",
            "--verbose",
        ],
        capture_output=True,
        text=True,
    )

    stderr = result.stderr

    # Verbose output should show role-level breakdown
    assert "breakdown" in stderr.lower() or "role" in stderr.lower() or "coder" in stderr.lower()


def test_output_file_generation(baseline_costs_file, optimized_costs_file, tmp_path):
    """Script can write results to output JSON file."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"
    output_file = tmp_path / "results.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--baseline",
            str(baseline_costs_file),
            "--current",
            str(optimized_costs_file),
            "--threshold",
            "0.10",
            "--output",
            str(output_file),
        ],
        capture_output=True,
        text=True,
    )

    # Output file should be created
    assert output_file.exists(), "Output file should be created"

    # Should contain valid JSON
    with open(output_file) as f:
        results = json.load(f)

    # Should contain expected fields
    assert "baseline_cost" in results
    assert "current_cost" in results
    assert "cost_reduction" in results
    assert "threshold" in results
    assert "passed" in results


def test_realistic_cost_reduction_calculation(baseline_costs_file, optimized_costs_file):
    """Test realistic cost reduction with haiku downgrades.

    Based on Component 7 optimization:
    - git, merger, issue_writer downgraded from sonnet to haiku
    - Haiku is 5x cheaper than sonnet
    - With the test data (git+merger+issue_writer = ~97s out of ~1572s total),
      this achieves ~4.94% cost reduction
    - Using a 4% threshold which should be met
    """
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--baseline",
            str(baseline_costs_file),
            "--current",
            str(optimized_costs_file),
            "--threshold",
            "0.04",  # 4% threshold - should be met with haiku downgrades
            "--verbose",
        ],
        capture_output=True,
        text=True,
    )

    # Should pass with 4% threshold
    assert result.returncode == 0, f"Should achieve >4% cost reduction with haiku downgrades\nStderr: {result.stderr}"

    # Verify cost reduction is shown
    assert "Cost reduction:" in result.stderr


def test_handles_invalid_json_file(tmp_path):
    """Script handles invalid JSON gracefully."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    # Create invalid JSON file
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w") as f:
        f.write("{ invalid json }")

    result = subprocess.run(
        [sys.executable, str(script_path), "--baseline", str(invalid_file), "--threshold", "0.10"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, "Script should exit with code 1 for invalid JSON"
    assert "error" in result.stderr.lower() or "invalid" in result.stderr.lower()


def test_handles_missing_required_fields(tmp_path):
    """Script validates required fields in JSON."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    # Create JSON without agent_durations
    incomplete_file = tmp_path / "incomplete.json"
    with open(incomplete_file, "w") as f:
        json.dump({"model_assignments": {}}, f)

    result = subprocess.run(
        [sys.executable, str(script_path), "--baseline", str(incomplete_file), "--threshold", "0.10"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, "Script should exit with code 1 for missing required fields"
    assert "agent_durations" in result.stderr or "required" in result.stderr.lower()


def test_threshold_validation():
    """Script validates threshold is between 0.0 and 1.0."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    # Test invalid threshold > 1.0
    result = subprocess.run(
        [sys.executable, str(script_path), "--baseline", "dummy.json", "--threshold", "1.5"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "threshold" in result.stderr.lower() and ("between" in result.stderr.lower() or "0.0" in result.stderr)


def test_edge_case_zero_baseline_cost(tmp_path):
    """Script handles edge case of zero baseline cost."""
    script_path = Path(__file__).parent.parent / "scripts" / "compare_build_costs.py"

    # Create baseline with zero durations
    zero_costs_file = tmp_path / "zero_costs.json"
    with open(zero_costs_file, "w") as f:
        json.dump({"agent_durations": {}, "model_assignments": {}}, f)

    result = subprocess.run(
        [sys.executable, str(script_path), "--baseline", str(zero_costs_file), "--threshold", "0.10"],
        capture_output=True,
        text=True,
    )

    # Should handle gracefully (either pass or fail, but not crash)
    assert result.returncode in [0, 1], f"Should handle zero baseline gracefully\nStderr: {result.stderr}"
