"""
Utility functions for the TLDR package.
"""

import re
import os
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()

# File patterns to ignore (binary and generated files)
IGNORE_PATTERNS = [
    r".*\.lock$",  # package-lock.json, yarn.lock, etc.
    r".*\.min\.js$",  # Minified JS
    r".*\.min\.css$",  # Minified CSS
    r".*\.woff2?$",  # Web fonts
    r".*\.ico$",  # Icons
    r".*\.png$",  # Images
    r".*\.jpe?g$",  # Images
    r".*\.gif$",  # Images
    r".*\.svg$",  # Vector images
    r".*\.pdf$",  # PDFs
    r".*\.zip$",  # Archives
    r".*\.tar\.gz$",  # Archives
    r".*\.map$",  # Source maps
    r"node_modules/.*",  # Node modules
    r"dist/.*",  # Distribution directories
    r"build/.*",  # Build directories
    r"vendor/.*",  # Vendor directories
    r".*-lock\.yaml$",  # Poetry lock files
    r".*\.pb\.go$",  # Protobuf generated files
    r".*_pb2\.py$",  # Protobuf generated Python
    r"Pipfile\.lock$",  # Pipfile lock
    r"Cargo\.lock$",  # Cargo lock
    r".*\.jar$",  # Java archives
    r".*\.exe$",  # Executables
    r".*\.dll$",  # DLLs
    r".*\.so$",  # Shared objects
    r".*\.dylib$",  # Dynamic libraries
]


def is_binary_file(content: str) -> bool:
    """Check if a file appears to be binary based on its content"""
    # Check for null bytes which typically indicate binary content
    if "\0" in content[:8000]:  # Check first 8KB
        return True

    # Check for high ratio of non-printable characters
    sample = content[:8000]
    if sample:
        printable_ratio = sum(c.isprintable() for c in sample) / len(sample)
        return printable_ratio < 0.8  # If less than 80% is printable, consider binary

    return False


def should_ignore_file(file_path: str) -> bool:
    """Check if a file should be ignored based on patterns and content"""
    # Check against patterns
    for pattern in IGNORE_PATTERNS:
        if re.match(pattern, file_path):
            console.print(
                f"[yellow]Ignoring file matching pattern '{pattern}': {file_path}[/yellow]"
            )
            return True

    return False


def find_git_root(start_path: Path) -> Path:
    """Find the root directory of the git repository"""
    try:
        result = subprocess.run(
            "git rev-parse --show-toplevel",
            shell=True,
            cwd=start_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        console.print(
            "[yellow]Warning: Not a git repository, using provided path[/yellow]"
        )
        return start_path


def run_command(repo_path: Path, command: str) -> str:
    """Run a shell command and return its output"""
    try:
        console.print(f"[blue]Running command:[/blue] {command}")
        result = subprocess.run(
            command,
            shell=True,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running command:[/red] {e}")
        return f"Error: {e.stderr}"


def get_openai_client():
    """Create and return an OpenAI client"""
    # Lazy import to avoid loading OpenAI when not needed
    from openai import AsyncOpenAI

    # Get API key from environment variable or use a default for testing
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print(
            "[yellow]Warning: OPENAI_API_KEY not found in environment variables[/yellow]"
        )
        console.print(
            "[yellow]Please set your OpenAI API key: export OPENAI_API_KEY='your-key'[/yellow]"
        )
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )

    return AsyncOpenAI(api_key=api_key)


# Add an async function to run commands
async def run_command_async(repo_path: Path, command: str) -> str:
    """Run a shell command asynchronously and return its output"""
    import asyncio

    try:
        console.print(f"[blue]Running command async:[/blue] {command}")
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_path,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            console.print(f"[red]Error running command:[/red] {stderr.decode()}")
            return f"Error: {stderr.decode()}"

        return stdout.decode().strip()
    except Exception as e:
        console.print(f"[red]Error running command:[/red] {e}")
        return f"Error: {e}"
