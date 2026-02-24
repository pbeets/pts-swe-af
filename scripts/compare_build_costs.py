#!/usr/bin/env python3
"""Build cost comparison and validation.

Compares baseline build costs with optimized build costs to validate:
- Cost reduction ≥15% from haiku model downgrades (Component 7)
- Used in integration tests to gate Component 7 deployment

The script expects cost data in JSON format with agent durations and model assignments.
Cost reduction is calculated based on relative pricing differences between models.

Usage:
    python scripts/compare_build_costs.py --baseline baseline_costs.json --threshold 0.15
    python scripts/compare_build_costs.py --baseline examples/baseline_costs.json --threshold 0.20 --current optimized_costs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Model cost multipliers (relative to haiku = 1.0)
# Based on typical Claude pricing: Sonnet is ~5x more expensive than Haiku
MODEL_COST_MULTIPLIERS = {
    "haiku": 1.0,
    "sonnet": 5.0,
    "opus": 15.0,  # For completeness, though not commonly used in this optimization
}


def load_costs_from_file(file_path: Path) -> dict[str, Any]:
    """Load cost data from JSON file.

    Args:
        file_path: Path to JSON file containing cost data

    Returns:
        Dict with agent_durations, model_assignments, and optionally total_cost

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If required fields are missing
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cost file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate required fields
    if "agent_durations" not in data:
        raise ValueError(f"Missing required field 'agent_durations' in {file_path}")

    # model_assignments is optional - if not present, assume all sonnet
    if "model_assignments" not in data:
        data["model_assignments"] = {}

    return data


def calculate_total_cost(
    agent_durations: dict[str, list[float]],
    model_assignments: dict[str, str],
    default_model: str = "sonnet"
) -> float:
    """Calculate total cost based on agent durations and model assignments.

    Cost is calculated as:
        total_cost = sum(duration * model_cost_multiplier for each agent call)

    Args:
        agent_durations: Dict mapping agent role to list of durations (seconds)
        model_assignments: Dict mapping agent role to model name
        default_model: Default model to use if role not in model_assignments

    Returns:
        Total cost (arbitrary units, relative to haiku = 1.0)
    """
    total_cost = 0.0

    for role, durations in agent_durations.items():
        # Get model for this role
        model = model_assignments.get(role, default_model)
        cost_multiplier = MODEL_COST_MULTIPLIERS.get(model, MODEL_COST_MULTIPLIERS["sonnet"])

        # Sum up costs for all calls of this role
        role_duration = sum(durations)
        role_cost = role_duration * cost_multiplier
        total_cost += role_cost

    return total_cost


def calculate_cost_reduction(baseline_cost: float, current_cost: float) -> float:
    """Calculate cost reduction percentage.

    Args:
        baseline_cost: Baseline total cost
        current_cost: Current/optimized total cost

    Returns:
        Cost reduction as fraction (e.g., 0.15 = 15% reduction)
        Returns 0.0 if baseline_cost is 0
    """
    if baseline_cost == 0.0:
        return 0.0

    reduction = (baseline_cost - current_cost) / baseline_cost
    return max(0.0, reduction)  # Clamp to non-negative


def format_cost_breakdown(
    agent_durations: dict[str, list[float]],
    model_assignments: dict[str, str],
    default_model: str = "sonnet",
    top_n: int = 5
) -> str:
    """Format cost breakdown by role.

    Args:
        agent_durations: Dict mapping agent role to list of durations
        model_assignments: Dict mapping agent role to model name
        default_model: Default model to use if role not in assignments
        top_n: Number of top cost contributors to show

    Returns:
        Formatted string with cost breakdown
    """
    role_costs = []

    for role, durations in agent_durations.items():
        model = model_assignments.get(role, default_model)
        cost_multiplier = MODEL_COST_MULTIPLIERS.get(model, MODEL_COST_MULTIPLIERS["sonnet"])
        role_duration = sum(durations)
        role_cost = role_duration * cost_multiplier
        num_calls = len(durations)

        role_costs.append({
            "role": role,
            "model": model,
            "duration": role_duration,
            "cost": role_cost,
            "calls": num_calls,
        })

    # Sort by cost descending
    role_costs.sort(key=lambda x: x["cost"], reverse=True)

    # Format top N
    lines = []
    for i, item in enumerate(role_costs[:top_n], 1):
        lines.append(
            f"    {i}. {item['role']:20s} ({item['model']:6s}): "
            f"{item['cost']:8.1f} cost units ({item['duration']:6.1f}s, {item['calls']} calls)"
        )

    total_cost = sum(item["cost"] for item in role_costs)
    lines.append(f"    {'Total':>29s}: {total_cost:8.1f} cost units")

    return "\n".join(lines)


def main():
    """Main entry point for build cost comparison."""
    parser = argparse.ArgumentParser(
        description="Compare build costs and validate cost reduction"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline costs JSON file",
    )
    parser.add_argument(
        "--current",
        type=str,
        help="Path to current/optimized costs JSON file (optional, defaults to stdin or baseline comparison)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Minimum cost reduction threshold as fraction (e.g., 0.15 = 15%% reduction required)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path for comparison results (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed cost breakdown",
    )

    args = parser.parse_args()

    # Validate threshold
    if args.threshold < 0.0 or args.threshold > 1.0:
        print("Error: --threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    try:
        # Load baseline costs
        baseline_path = Path(args.baseline)
        print(f"Loading baseline costs from: {args.baseline}", file=sys.stderr)
        baseline_data = load_costs_from_file(baseline_path)

        # Load current costs
        if args.current:
            current_path = Path(args.current)
            print(f"Loading current costs from: {args.current}", file=sys.stderr)
            current_data = load_costs_from_file(current_path)
        else:
            # If no current file specified, use baseline data but with optimized model assignments
            # This allows testing the script with a single baseline file
            print("No --current specified, using baseline data", file=sys.stderr)
            current_data = baseline_data

        # Calculate baseline cost (all agents using sonnet by default)
        baseline_cost = calculate_total_cost(
            baseline_data["agent_durations"],
            baseline_data.get("model_assignments", {}),
            default_model="sonnet"
        )

        # Calculate current cost (with optimized model assignments)
        current_cost = calculate_total_cost(
            current_data["agent_durations"],
            current_data.get("model_assignments", {}),
            default_model="sonnet"
        )

        # Calculate cost reduction
        cost_reduction = calculate_cost_reduction(baseline_cost, current_cost)

        # Prepare results
        results = {
            "baseline_cost": baseline_cost,
            "current_cost": current_cost,
            "cost_reduction": cost_reduction,
            "threshold": args.threshold,
            "passed": cost_reduction >= args.threshold,
        }

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Results written to: {args.output}", file=sys.stderr)

        # Print comparison summary
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Build Cost Comparison", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"  Baseline cost:       {baseline_cost:10.1f} cost units", file=sys.stderr)
        print(f"  Current cost:        {current_cost:10.1f} cost units", file=sys.stderr)
        print(f"  Cost reduction:      {cost_reduction:10.2%} ({baseline_cost - current_cost:.1f} cost units)", file=sys.stderr)
        print(f"  Threshold:           {args.threshold:10.2%}", file=sys.stderr)
        print(f"", file=sys.stderr)

        # Show detailed breakdown if verbose
        if args.verbose:
            print(f"Baseline Cost Breakdown (top 5 roles):", file=sys.stderr)
            breakdown_baseline = format_cost_breakdown(
                baseline_data["agent_durations"],
                baseline_data.get("model_assignments", {}),
                default_model="sonnet"
            )
            print(breakdown_baseline, file=sys.stderr)
            print(f"", file=sys.stderr)

            print(f"Current Cost Breakdown (top 5 roles):", file=sys.stderr)
            breakdown_current = format_cost_breakdown(
                current_data["agent_durations"],
                current_data.get("model_assignments", {}),
                default_model="sonnet"
            )
            print(breakdown_current, file=sys.stderr)
            print(f"", file=sys.stderr)

        # Determine status
        if cost_reduction >= args.threshold:
            print(f"✓ PASS: Cost reduction {cost_reduction:.2%} >= threshold {args.threshold:.2%}", file=sys.stderr)
            print(f"  Component 7 (Model Downgrade) APPROVED for deployment", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            sys.exit(0)
        else:
            print(f"✗ FAIL: Cost reduction {cost_reduction:.2%} < threshold {args.threshold:.2%}", file=sys.stderr)
            print(f"  Component 7 (Model Downgrade) REJECTED - cost savings insufficient", file=sys.stderr)
            shortfall = args.threshold - cost_reduction
            print(f"  Shortfall: {shortfall:.2%} ({shortfall * baseline_cost:.1f} cost units)", file=sys.stderr)
            print(f"  ACTION: Review model assignments or adjust threshold", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
