"""Prompt builder for the fast .ai() review sanity gate.

This replaces the full .harness() code reviewer on the default (non-flagged)
path.  It's a lightweight single-shot check that catches obvious breaks
(compilation errors, missing files, test failures) without multi-turn tool use.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a fast sanity-check gate in an autonomous coding pipeline. \
A coder agent has just implemented changes for an issue. Your job is to \
quickly assess whether the changes are likely clean or need attention.

You do NOT have tool access. You make your decision based solely on the \
coder's self-reported results and the issue's acceptance criteria.

## Decision Rules

### approve
- Coder reports tests_passed=true
- Files changed look reasonable for the issue scope
- Summary indicates all acceptance criteria were addressed
- No red flags in the coder's summary

### fix
- Coder reports tests_passed=false
- Coder summary mentions incomplete work or known issues
- Files changed seem insufficient for the acceptance criteria
- Minor concerns that the coder should address

### block
- Coder reports a fundamental failure (crash, missing dependency, wrong approach)
- Security red flags mentioned in the summary
- Coder explicitly says the approach won't work

## Confidence

Set confident=false when:
- The issue is complex and you can't assess from the summary alone
- The coder's summary is vague or contradictory
- You're unsure whether tests_passed=true is trustworthy

When confident=false, the pipeline will fall back to a full code review.\
"""


def review_sanity_gate_task_prompt(
    coder_result: dict,
    issue: dict,
    iteration_id: str = "",
) -> str:
    """Build the task prompt for the review sanity gate.

    Args:
        coder_result: CoderResult dict with files_changed, summary, tests_passed.
        issue: The issue dict (name, title, acceptance_criteria, etc.)
        iteration_id: UUID for this iteration's artifact tracking.
    """
    sections: list[str] = []

    sections.append("## Issue")
    sections.append(f"- **Name**: {issue.get('name', '(unknown)')}")
    sections.append(f"- **Title**: {issue.get('title', '(unknown)')}")

    ac = issue.get("acceptance_criteria", [])
    if ac:
        sections.append("- **Acceptance Criteria**:")
        sections.extend(f"  - {c}" for c in ac)

    sections.append(f"\n## Coder's Changes")
    sections.append(f"- **Summary**: {coder_result.get('summary', '(none)')}")

    tests_passed = coder_result.get("tests_passed")
    test_summary = coder_result.get("test_summary", "")
    sections.append(f"- **tests_passed**: {tests_passed}")
    if test_summary:
        sections.append(f"- **test_summary**: {test_summary}")

    files = coder_result.get("files_changed", [])
    if files:
        sections.append("- **Files changed**:")
        sections.extend(f"  - `{f}`" for f in files)
    else:
        sections.append("- **Files changed**: (none reported)")

    sections.append(
        "\n## Your Task\n"
        "Quickly assess this iteration. Decide: approve, fix, or block.\n"
        "List any risk_areas (empty list if clean).\n"
        "Set confident=false if you can't reliably assess from this information alone."
    )

    return "\n".join(sections)
