"""Prompt builder for the replanner triage gate (.ai() classifier).

This is a fast pre-filter that classifies failures before invoking the
expensive full replanner harness. It determines whether DAG restructuring
is actually needed or a cheaper action (skip downstream, retry) suffices.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a failure triage classifier in a fully autonomous software \
engineering pipeline. You receive a list of failed issues and a brief DAG \
state summary, and your job is to quickly classify the failure and \
recommend the cheapest adequate response.

## Failure Types

### transient
Network timeouts, resource exhaustion, rate limits, flaky test runs. \
These are NOT bugs in the plan or code — they are infrastructure hiccups.

### fixable
Missing import, syntax error, wrong file path, typo in config. The code \
change itself is small and obvious. Note: the issue advisor already handles \
retries for fixable errors, so if the failure reached you it likely means \
retries were exhausted. Recommend skip_downstream.

### structural
Wrong decomposition, missing dependency between issues, architectural \
mismatch, issue scope too large or too small, circular dependency in the \
plan. The DAG itself needs restructuring.

### environmental
Missing tool/binary, wrong runtime version, missing system dependency, \
Docker not running, permissions issue. The code and plan are fine but the \
environment cannot execute them.

## Recommended Actions

- **continue**: Proceed with remaining issues (ignore this failure)
- **retry**: Retry the failed issues (only if transient)
- **skip_downstream**: Mark failed issues and their dependents as skipped
- **replan**: Invoke the full replanner to restructure the DAG

## Decision Rules

1. Only recommend **replan** when the DAG structure itself is wrong — \
wrong issue boundaries, missing dependencies, architectural errors.
2. For transient/environmental/fixable failures, prefer \
**skip_downstream** — it is 10-20x cheaper than replanning.
3. Set **confident = false** if the error messages are ambiguous or you \
cannot clearly classify the failure. When in doubt, let the full \
replanner decide.
4. Be conservative: a missed replan costs one more failed level; an \
unnecessary replan wastes significant budget.

## Output

Return your classification in the structured schema. Keep reasoning to \
1-2 sentences.\
"""


def replanner_triage_gate_task_prompt(
    failed_issues: list[dict],
    dag_state_summary: dict,
) -> str:
    """Build the task prompt for the replanner triage gate.

    Args:
        failed_issues: List of dicts with issue_name, error_message,
            error_context, outcome, attempts.
        dag_state_summary: Brief dict with total_issues, completed,
            failed, current_level, replan_count.
    """
    sections: list[str] = []

    # DAG state summary
    sections.append("## DAG State")
    sections.append(
        f"- Total issues: {dag_state_summary.get('total_issues', '?')}"
    )
    sections.append(
        f"- Completed: {dag_state_summary.get('completed', '?')}"
    )
    sections.append(f"- Failed: {dag_state_summary.get('failed', '?')}")
    sections.append(
        f"- Current level: {dag_state_summary.get('current_level', '?')}"
    )
    sections.append(
        f"- Previous replans: {dag_state_summary.get('replan_count', 0)}"
    )

    # Failed issues
    sections.append(f"\n## Failed Issues ({len(failed_issues)})")
    for issue in failed_issues:
        name = issue.get("issue_name", "unknown")
        outcome = issue.get("outcome", "unknown")
        attempts = issue.get("attempts", 1)
        error_msg = issue.get("error_message", "(no error message)")
        error_ctx = issue.get("error_context", "")

        sections.append(f"\n### {name}")
        sections.append(f"- **Outcome**: {outcome}")
        sections.append(f"- **Attempts**: {attempts}")
        sections.append(f"- **Error**: {error_msg}")
        if error_ctx:
            # Truncate long error context to keep prompt concise
            truncated = error_ctx[:500]
            if len(error_ctx) > 500:
                truncated += "... (truncated)"
            sections.append(f"- **Context**: {truncated}")

    sections.append(
        "\n## Your Task\n"
        "Classify the failure type and recommend an action. "
        "Only recommend 'replan' if the DAG structure itself is wrong."
    )

    return "\n".join(sections)
