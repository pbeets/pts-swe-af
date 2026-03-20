"""Prompt builder for the post-merge batch reviewer (.harness()).

This runs AFTER all issues in a level have been merged into the integration
branch. It sees the full combined diff and catches cross-issue quality issues
that per-iteration reviews cannot detect: naming inconsistencies, duplicated
code across issues, architectural violations, import conflicts, etc.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a senior engineer performing a thorough batch code review after \
multiple issues have been merged into an integration branch. Unlike \
per-iteration reviews that see one issue at a time, you see the COMBINED \
diff of all merged changes.

## Your Unique Value

You catch problems that are INVISIBLE to per-issue reviewers:
- **Cross-issue naming inconsistencies**: same concept named differently
- **Duplicated code**: two issues independently implemented similar utilities
- **Import conflicts**: circular imports or conflicting module structures
- **Architectural violations**: patterns that only emerge in the combined view
- **Integration gaps**: interfaces that don't match between issues
- **Test isolation**: tests from one issue breaking another's assumptions

## Severity Classification

### BLOCKING (blocking_issues)
Only for issues that MUST be fixed:
- Circular imports that crash at runtime
- Two issues writing conflicting code to the same file
- Security vulnerabilities from combined changes
- Data corruption from interacting code paths

### CROSS-ISSUE CONCERNS (cross_issue_concerns)
Important but non-blocking observations:
- Naming inconsistencies across issues
- Duplicated utility code
- Missed shared abstractions
- Test coverage gaps in interaction points

## Decision Rules

- If no blocking issues → approved = true
- Blocking issues should be rare (per-issue reviews already caught per-issue bugs)
- Focus on what's ONLY visible in the combined diff
- Don't re-review individual issues — that already happened

## Tools Available

You have full read access to the merged codebase:
- READ files to inspect the combined source
- GLOB to find files by pattern
- GREP to search for patterns across the merged codebase
- BASH to run tests on the merged code

Do NOT modify source files.\
"""


def batch_reviewer_task_prompt(
    repo_path: str,
    integration_branch: str,
    completed_issues: list[dict],
    prd_summary: str = "",
    architecture_summary: str = "",
) -> str:
    """Build the task prompt for the batch reviewer.

    Args:
        repo_path: Absolute path to the repository.
        integration_branch: The integration branch with merged changes.
        completed_issues: List of dicts with issue_name, files_changed, summary.
        prd_summary: Brief PRD summary for context.
        architecture_summary: Brief architecture summary for context.
    """
    sections: list[str] = []

    sections.append(f"## Repository\n`{repo_path}` on branch `{integration_branch}`")

    if prd_summary:
        sections.append(f"\n## PRD Summary\n{prd_summary}")
    if architecture_summary:
        sections.append(f"\n## Architecture Summary\n{architecture_summary}")

    sections.append(f"\n## Merged Issues ({len(completed_issues)})")
    for issue in completed_issues:
        name = issue.get("issue_name", issue.get("name", "(unknown)"))
        summary = issue.get("summary", issue.get("result_summary", "(no summary)"))
        files = issue.get("files_changed", [])
        sections.append(f"\n### {name}")
        sections.append(f"- **Summary**: {summary}")
        if files:
            sections.append("- **Files changed**:")
            sections.extend(f"  - `{f}`" for f in files)

    # Identify files touched by multiple issues
    file_to_issues: dict[str, list[str]] = {}
    for issue in completed_issues:
        name = issue.get("issue_name", issue.get("name", "?"))
        for f in issue.get("files_changed", []):
            file_to_issues.setdefault(f, []).append(name)
    shared_files = {f: issues for f, issues in file_to_issues.items() if len(issues) > 1}
    if shared_files:
        sections.append("\n## Files Modified by Multiple Issues (high-risk)")
        for f, issues in sorted(shared_files.items()):
            sections.append(f"- `{f}` — modified by: {', '.join(issues)}")

    sections.append(
        "\n## Your Task\n"
        "1. Review the COMBINED diff on the integration branch.\n"
        "   Run: `git diff HEAD~N...HEAD` or read the changed files directly.\n"
        "2. Focus on CROSS-ISSUE concerns:\n"
        "   - Naming inconsistencies between issues\n"
        "   - Duplicated code or utilities\n"
        "   - Circular imports or conflicting module structures\n"
        "   - Interface mismatches between issues\n"
        "   - Files modified by multiple issues (listed above if any)\n"
        "3. Run the test suite to verify nothing broke in integration.\n"
        "4. Report: approved (bool), blocking_issues, cross_issue_concerns, summary."
    )

    return "\n".join(sections)
