"""
Git and GitHub PR interaction functionality.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, TypedDict

from rich.console import Console

from tldr.utils import run_command, should_ignore_file, is_binary_file

console = Console()


class PRDiff(TypedDict):
    path: str
    content: str


class PRInfo(TypedDict):
    title: str
    number: int
    body: str
    link: str
    base_branch: str
    head_branch: str
    diffs: List[PRDiff]


def get_current_pr(repo_path: Path) -> Optional[PRInfo]:
    """Get information about the current PR with specific typed fields"""
    try:
        # Check if we're in a PR branch
        pr_json = run_command(
            repo_path,
            "gh pr view --json title,number,body,url,files,baseRefName,headRefName",
        )
        pr_data = json.loads(pr_json)

        # Debug the response
        console.print(f"[blue]PR data has {len(pr_data.get('files', []))} files[/blue]")

        # Extract the diffs for each changed file
        diffs = []
        if "files" in pr_data and "baseRefName" in pr_data:
            base_branch = pr_data["baseRefName"]

            for file_info in pr_data["files"]:
                file_path = file_info["path"]
                try:
                    # Use the PR's base branch to get the correct diff
                    diff_content = run_command(
                        repo_path,
                        f'git diff origin/{base_branch} -- "{file_path}"',
                    )

                    if diff_content.strip():
                        diffs.append({"path": file_path, "content": diff_content})
                        console.print(
                            f"[green]Got diff for {file_path} ({len(diff_content)} chars)[/green]"
                        )
                    else:
                        console.print(f"[yellow]Empty diff for {file_path}[/yellow]")
                except Exception as e:
                    console.print(
                        f"[yellow]Could not get diff for {file_path}: {e}[/yellow]"
                    )

        # If no diffs found with the first method, try a different approach
        if not diffs:
            console.print(
                "[yellow]No diffs found with base branch method, trying alternative...[/yellow]"
            )

            # Try to get files from GitHub API
            files_list = pr_data.get("files", [])
            for file_info in files_list:
                file_path = file_info["path"]
                try:
                    # Try to get diff directly from the current branch
                    diff_content = run_command(
                        repo_path,
                        f'git diff HEAD^ -- "{file_path}"',
                    )

                    if diff_content.strip():
                        diffs.append({"path": file_path, "content": diff_content})
                        console.print(
                            f"[green]Got diff for {file_path} with alternative method[/green]"
                        )
                    else:
                        console.print(
                            f"[yellow]Empty diff for {file_path} with alternative method[/yellow]"
                        )
                except Exception as e:
                    console.print(
                        f"[yellow]Alternative diff method failed for {file_path}: {e}[/yellow]"
                    )

        console.print(f"[blue]Found {len(diffs)} diffs in total[/blue]")

        # Create the typed return
        return {
            "title": pr_data.get("title", ""),
            "number": pr_data.get("number", 0),
            "body": pr_data.get("body", ""),
            "link": pr_data.get("url", ""),
            "base_branch": pr_data.get("baseRefName", ""),
            "head_branch": pr_data.get("headRefName", ""),
            "diffs": diffs,
        }
    except Exception as e:
        console.print(f"[red]Error getting PR info:[/red] {e}")
        return None


def process_pr_files(
    repo_path: Path,
    pr_info: PRInfo,
) -> Dict[str, Any]:
    """Process PR files, filtering out binary and ignored files"""
    # Extract file information from diffs
    diffs = pr_info.get("diffs", [])
    processed_files = []
    ignored_files = []
    binary_files = []

    for diff_item in diffs:
        file_path = diff_item["path"]

        if should_ignore_file(file_path):
            ignored_files.append(file_path)
            continue

        try:
            # Try to check file content to detect binary files
            try:
                file_sample = run_command(
                    repo_path, f"head -c 8000 {file_path} 2>/dev/null"
                )
                if is_binary_file(file_sample):
                    binary_files.append(file_path)
                    continue
            except Exception:
                # If we can't check content, proceed with the file
                pass

            # Get additions and deletions from diff content
            diff_content = diff_item["content"]
            additions = diff_content.count("\n+") - diff_content.count("\n+++")
            deletions = diff_content.count("\n-") - diff_content.count("\n---")

            # Create file info object similar to what GitHub API would return
            processed_files.append(
                {"path": file_path, "additions": additions, "deletions": deletions}
            )
        except Exception as e:
            console.print(
                f"[yellow]Warning: Error processing file {file_path}: {e}[/yellow]"
            )

    # Log ignored and binary files
    if ignored_files:
        console.print(
            f"[yellow]Ignoring {len(ignored_files)} files based on patterns:[/yellow]"
        )
        for f in ignored_files[:5]:  # Show only the first 5 to avoid clutter
            console.print(f"  - {f}")
        if len(ignored_files) > 5:
            console.print(f"  - ... and {len(ignored_files) - 5} more")

    if binary_files:
        console.print(
            f"[yellow]Skipping {len(binary_files)} detected binary files:[/yellow]"
        )
        for f in binary_files[:5]:  # Show only the first 5
            console.print(f"  - {f}")
        if len(binary_files) > 5:
            console.print(f"  - ... and {len(binary_files) - 5} more")

    return {
        "processed_files": processed_files,
        "ignored_files": ignored_files,
        "binary_files": binary_files,
    }
