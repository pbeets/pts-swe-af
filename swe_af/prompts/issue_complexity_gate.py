"""Prompt builder for the Issue Complexity Gate — a fast .ai() classifier."""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are a fast issue complexity classifier in an autonomous coding pipeline. \
You receive a single issue specification and classify it into one of three \
complexity tiers so the pipeline can choose the right execution path.

## Complexity Tiers

### trivial
- Rename, typo fix, config change, single-line edit
- No new files, no new logic, no tests needed
- Confidence: only classify as trivial if you are very sure

### standard
- Typical feature or bug fix touching 1-3 files
- May need tests but scope is clear and bounded
- This is the DEFAULT — use it when unsure

### complex
- Touches 4+ files, cross-cutting concerns, new subsystem
- Multiple acceptance criteria with interdependencies
- Requires deeper QA (parallel QA + reviewer + synthesizer path)

## needs_qa Decision

Set `needs_qa = true` when:
- The issue is "complex"
- The issue modifies public APIs or shared interfaces
- The acceptance criteria include explicit test requirements
- Multiple files are created AND modified

Set `needs_qa = false` when:
- The issue is "trivial"
- The issue is "standard" with clear, bounded scope

## Confidence

Set `confident = true` when the issue description gives you enough signal \
to classify reliably. Set `confident = false` when the description is vague, \
ambiguous, or missing key details — the pipeline will fall back to static \
guidance in that case.

When in doubt, classify as "standard" with needs_qa=false and confident=false.\
"""


def issue_complexity_gate_task_prompt(
    issue: dict,
) -> str:
    """Build the task prompt for the issue complexity gate.

    Args:
        issue: Issue dict with name, description, acceptance_criteria,
               files_to_create, files_to_modify, guidance.
    """
    sections: list[str] = []

    sections.append("## Issue to Classify")
    sections.append(f"- **Name**: {issue.get('name', '?')}")

    description = issue.get("description", "")
    if description:
        sections.append(f"- **Description**: {description[:500]}")

    ac = issue.get("acceptance_criteria", [])
    if ac:
        sections.append(f"- **Acceptance Criteria** ({len(ac)} total):")
        for criterion in ac[:8]:
            sections.append(f"  - {criterion}")
        if len(ac) > 8:
            sections.append(f"  - ... and {len(ac) - 8} more")

    files_create = issue.get("files_to_create", [])
    if files_create:
        sections.append(f"- **Files to create**: {len(files_create)} — {', '.join(files_create[:5])}")

    files_modify = issue.get("files_to_modify", [])
    if files_modify:
        sections.append(f"- **Files to modify**: {len(files_modify)} — {', '.join(files_modify[:5])}")

    guidance = issue.get("guidance") or {}
    if guidance:
        sections.append(f"- **Guidance**: {str(guidance)[:300]}")

    sections.append(
        "\n## Your Task\n"
        "Classify this issue's complexity as trivial, standard, or complex.\n"
        "Decide whether it needs the deeper QA path (needs_qa).\n"
        "Report your confidence in the classification."
    )

    return "\n".join(sections)
