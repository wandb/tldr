"""
Git and GitHub PR interaction functionality.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from rich.console import Console

from tldr.utils import run_command, should_ignore_file, is_binary_file

console = Console()


def get_current_pr(repo_path: Path) -> Optional[Dict[str, Any]]:
    """Get information about the current PR"""
    try:
        # Check if we're in a PR branch
        pr_json = run_command(
            repo_path,
            "gh pr view --json title,number,body,files,additions,deletions,baseRefName",
        )
        return json.loads(pr_json)
    except Exception as e:
        console.print(f"[red]Error getting PR info:[/red] {e}")
        return None


def process_pr_files(
    repo_path: Path,
    pr_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Process PR files, filtering out binary and ignored files"""
    changed_files = pr_info.get("files", [])

    # Filter out binary and ignored files
    processed_files = []
    ignored_files = []
    binary_files = []

    for file in changed_files:
        file_path = file["path"]
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

            processed_files.append(file)
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


def get_file_diffs(
    repo_path: Path,
    pr_info: Dict[str, Any],
    processed_files: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Get diffs for all processed files in the PR"""
    diffs = {}

    # Get diffs for all files
    for file in processed_files:
        # Get the full diff
        diff = run_command(
            repo_path, f"git diff origin/{pr_info['baseRefName']} -- {file['path']}"
        )

        # Store the diff
        diffs[file["path"]] = diff

    return diffs
