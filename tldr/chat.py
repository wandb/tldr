"""
Interactive chat functionality for PR exploration.
"""

import os
import weave
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from tldr.utils import (
    run_command,
    should_ignore_file,
    is_binary_file,
    get_openai_client,
)
from tldr.agent import extract_relevant_code, summarize_grep_results
from tldr.git import get_current_pr, process_pr_files

console = Console()


@weave.op()
def chat_loop(
    repo_path: Path,
    model: str,
    initial_summary: str = None,
) -> str:
    """Run an interactive chat loop about the PR"""
    # Get PR metadata and initial context
    pr_info = get_current_pr(repo_path)
    if not pr_info:
        return "No active PR found in this repository"

    # Initialize context
    context = []

    # Add basic PR information to context
    context.append(
        {
            "role": "system",
            "content": "You are a helpful assistant that can answer questions about a GitHub Pull Request.",
        }
    )

    context.append(
        {
            "role": "user",
            "content": f"PR #{pr_info['number']}: {pr_info['title']}\n\nDescription: {pr_info['body']}\nBranches: {pr_info['head_branch']} → {pr_info['base_branch']}",
        }
    )

    # Process files information
    files_info = process_pr_files(repo_path, pr_info)
    processed_files = files_info["processed_files"]
    files_list = "\n".join(
        [
            f"- {f['path']} ({f['additions']} additions, {f['deletions']} deletions)"
            for f in processed_files
        ]
    )
    context.append({"role": "user", "content": f"Changed files:\n{files_list}"})

    # Keep track of known information to avoid duplicating work
    known_files = set(f["path"] for f in pr_info.get("files", []))
    explored_patterns = set()
    explored_files = set()  # Track files we've already added to context

    # Add the initial summary if provided
    if initial_summary:
        context.append({"role": "assistant", "content": initial_summary})

    console.print(
        "\n[bold blue]Chat mode enabled. Type 'exit' to quit or 'summarize' to generate a new summary.[/bold blue]"
    )
    console.print(
        "[bold blue]Ask questions about the PR or request more information.[/bold blue]\n"
    )
    console.print("[bold blue]Special commands:[/bold blue]")
    console.print("  [blue]exit[/blue] - Exit chat mode")
    console.print(
        "  [blue]summarize[/blue] - Generate a new summary based on current context"
    )
    console.print("  [blue]ls <directory>[/blue] - List files in a directory")
    console.print("  [blue]cat <file>[/blue] - Read the contents of a file")

    # Display the initial summary again in chat mode for reference
    if initial_summary:
        console.print("[bold green]Initial PR Summary (for reference):[/bold green]")
        console.print(Panel(Markdown(initial_summary), border_style="green"))

    while True:
        # Get user input
        user_input = typer.prompt("You")

        if user_input.lower() == "exit":
            break

        if user_input.lower() == "summarize":
            console.print("[bold blue]Generating new summary...[/bold blue]")
            # Add specific instruction to generate a summary
            temp_context = context.copy()
            temp_context.append(
                {
                    "role": "user",
                    "content": "Based on all the information so far, please provide a comprehensive summary of this PR.",
                }
            )

            client = get_openai_client()
            response = client.chat.completions.create(
                model=model,
                messages=temp_context,
                temperature=0.1,
            )
            new_summary = response.choices[0].message.content

            # Display the summary
            console.print("\n[bold green]Updated PR Summary:[/bold green]")
            console.print(Panel(Markdown(new_summary), border_style="green"))
            continue

        # Handle special command: ls - list directory contents
        if user_input.lower().startswith("ls "):
            try:
                dir_path = user_input[3:].strip()
                if not os.path.isabs(dir_path):
                    dir_path = os.path.join(repo_path, dir_path)

                # Run ls command
                result = run_command(repo_path, f"ls -la {dir_path}")

                # Add to context
                context.append(
                    {"role": "user", "content": f"List files in directory {dir_path}"}
                )
                context.append(
                    {
                        "role": "assistant",
                        "content": f"Contents of {dir_path}:\n```\n{result}\n```",
                    }
                )

                # Display to user
                console.print("\n[bold green]Directory Contents:[/bold green]")
                console.print(Panel(result, border_style="green"))
                continue
            except Exception as e:
                console.print(f"[red]Error listing directory: {e}[/red]")
                context.append(
                    {"role": "user", "content": f"List files in directory {dir_path}"}
                )
                context.append(
                    {
                        "role": "assistant",
                        "content": f"Error listing directory {dir_path}: {e}",
                    }
                )
                continue

        # Handle special command: cat - read file contents
        if user_input.lower().startswith("cat "):
            try:
                file_path = user_input[4:].strip()
                if not os.path.isabs(file_path):
                    file_path = os.path.join(repo_path, file_path)

                # Check if it's a binary file
                file_sample = run_command(
                    repo_path, f"head -c 8000 {file_path} 2>/dev/null"
                )
                if is_binary_file(file_sample):
                    message = f"The file {file_path} appears to be a binary file, so I can't provide its contents directly."
                    console.print(f"[yellow]{message}[/yellow]")
                    context.append(
                        {
                            "role": "user",
                            "content": f"Show contents of file {file_path}",
                        }
                    )
                    context.append({"role": "assistant", "content": message})
                else:
                    # Get file content
                    file_content = run_command(repo_path, f"cat {file_path}")

                    # Add to context, but keep within reasonable size
                    if len(file_content) > 5000:
                        # Extract relevant portions of the file using LLM
                        console.print(
                            "[blue]File is large, extracting most relevant parts...[/blue]"
                        )
                        relevant_content = extract_relevant_code(
                            file_content, "", model
                        )
                        context.append(
                            {
                                "role": "user",
                                "content": f"Show contents of file {file_path}",
                            }
                        )
                        context.append(
                            {
                                "role": "assistant",
                                "content": f"Contents of {file_path} (relevant parts):\n```\n{relevant_content}\n```",
                            }
                        )
                        console.print(
                            f"[green]Added relevant parts of {file_path} to context[/green]"
                        )
                    else:
                        context.append(
                            {
                                "role": "user",
                                "content": f"Show contents of file {file_path}",
                            }
                        )
                        context.append(
                            {
                                "role": "assistant",
                                "content": f"Contents of {file_path}:\n```\n{file_content}\n```",
                            }
                        )

                # Mark this file as explored
                explored_files.add(file_path)

                # Display to user that we've added the file
                console.print(f"[green]Added file {file_path} to context[/green]")
                continue
            except Exception as e:
                console.print(f"[red]Error reading file: {e}[/red]")
                context.append(
                    {"role": "user", "content": f"Show contents of file {file_path}"}
                )
                context.append(
                    {
                        "role": "assistant",
                        "content": f"Error reading file {file_path}: {e}",
                    }
                )
                continue

        # Check if user is asking about a specific file not yet explored
        for file_path in known_files:
            if (
                file_path.lower() in user_input.lower()
                and not should_ignore_file(file_path)
                and file_path not in explored_files
            ):
                try:
                    # Check if we're dealing with a binary file
                    try:
                        file_sample = run_command(
                            repo_path, f"head -c 8000 {file_path} 2>/dev/null"
                        )
                        if is_binary_file(file_sample):
                            context.append(
                                {
                                    "role": "user",
                                    "content": f"Give me information about the binary file {file_path}",
                                }
                            )
                            context.append(
                                {
                                    "role": "assistant",
                                    "content": f"The file {file_path} appears to be a binary file, so I can't provide its contents directly.",
                                }
                            )
                            explored_files.add(file_path)
                            continue
                    except Exception:
                        pass

                    # Get the file content if not already examined
                    file_content = run_command(repo_path, f"cat {file_path}")
                    diff = run_command(
                        repo_path,
                        f"git diff origin/{pr_info['baseRefName']} -- {file_path}",
                    )

                    # Add file information to context
                    context.append(
                        {
                            "role": "user",
                            "content": f"I need more information about the file {file_path}",
                        }
                    )
                    context.append(
                        {
                            "role": "assistant",
                            "content": f"I'll add information about {file_path} to our conversation.",
                        }
                    )
                    context.append(
                        {
                            "role": "user",
                            "content": f"Diff for {file_path}:\n```\n{diff[:2000]}{'...' if len(diff) > 2000 else ''}\n```",
                        }
                    )

                    # Add relevant code portions
                    relevant_code = extract_relevant_code(file_content, diff, model)
                    context.append(
                        {
                            "role": "user",
                            "content": f"Relevant code from {file_path}:\n```\n{relevant_code}\n```",
                        }
                    )

                    # Mark as explored
                    explored_files.add(file_path)
                    console.print(f"[green]Added file {file_path} to context[/green]")
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Couldn't add file information for {file_path}: {e}[/yellow]"
                    )

        # Check if user is asking about reading a file that's not in the PR
        if (
            "file" in user_input.lower()
            or "read" in user_input.lower()
            or "show" in user_input.lower()
            or "look at" in user_input.lower()
        ):
            # Try to extract file paths from the query
            client = get_openai_client()
            extract_messages = [
                {
                    "role": "system",
                    "content": "Extract file paths from the user query. Return only the paths, one per line. If no specific paths are mentioned but user wants to see a file, return 'NEEDS_MORE_INFO'.",
                },
                {
                    "role": "user",
                    "content": f"Extract file paths from this query: {user_input}",
                },
            ]

            extract_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=extract_messages,
                temperature=0.1,
                max_tokens=200,
            )

            potential_paths = [
                p.strip()
                for p in extract_response.choices[0].message.content.split("\n")
                if p.strip()
            ]

            # Check if we need more information
            if len(potential_paths) == 1 and potential_paths[0] == "NEEDS_MORE_INFO":
                # Skip this section and let the regular LLM handle it
                pass
            else:
                for path in potential_paths:
                    if path in explored_files:
                        continue

                    try:
                        # Check if this is a valid path
                        full_path = path
                        if not os.path.isabs(path):
                            full_path = os.path.join(repo_path, path)

                        if not os.path.exists(full_path):
                            continue

                        if os.path.isdir(full_path):
                            # List directory contents
                            result = run_command(repo_path, f"ls -la {full_path}")
                            context.append(
                                {
                                    "role": "user",
                                    "content": f"List files in directory {path}",
                                }
                            )
                            context.append(
                                {
                                    "role": "assistant",
                                    "content": f"Contents of {path}:\n```\n{result}\n```",
                                }
                            )
                            console.print(
                                f"[green]Added directory listing for {path} to context[/green]"
                            )
                        else:
                            # Check if it's a binary file
                            file_sample = run_command(
                                repo_path, f"head -c 8000 {full_path} 2>/dev/null"
                            )
                            if is_binary_file(file_sample):
                                message = f"The file {path} appears to be a binary file, so I can't provide its contents directly."
                                context.append(
                                    {
                                        "role": "user",
                                        "content": f"Show contents of file {path}",
                                    }
                                )
                                context.append(
                                    {"role": "assistant", "content": message}
                                )
                            else:
                                # Get file content
                                file_content = run_command(
                                    repo_path, f"cat {full_path}"
                                )

                                # Add to context, but keep within reasonable size
                                if len(file_content) > 5000:
                                    # Extract relevant portions of the file using LLM
                                    console.print(
                                        "[blue]File is large, extracting most relevant parts...[/blue]"
                                    )
                                    relevant_content = extract_relevant_code(
                                        file_content, "", model
                                    )
                                    context.append(
                                        {
                                            "role": "user",
                                            "content": f"Show contents of file {path}",
                                        }
                                    )
                                    context.append(
                                        {
                                            "role": "assistant",
                                            "content": f"Contents of {path} (relevant parts):\n```\n{relevant_content}\n```",
                                        }
                                    )
                                else:
                                    context.append(
                                        {
                                            "role": "user",
                                            "content": f"Show contents of file {path}",
                                        }
                                    )
                                    context.append(
                                        {
                                            "role": "assistant",
                                            "content": f"Contents of {path}:\n```\n{file_content}\n```",
                                        }
                                    )

                                # Mark this file as explored
                                explored_files.add(path)
                                console.print(
                                    f"[green]Added file {path} to context[/green]"
                                )
                    except Exception:
                        # Just skip this path if there's an error
                        pass

        # Add user's question to context
        context.append({"role": "user", "content": user_input})

        # Check if user is requesting grep search for specific patterns
        if (
            "search for" in user_input.lower()
            or "find" in user_input.lower()
            or "where is" in user_input.lower()
        ):
            # Extract potential search patterns
            client = get_openai_client()
            pattern_messages = [
                {
                    "role": "system",
                    "content": "Extract search patterns from the user query. Return only the patterns, one per line.",
                },
                {
                    "role": "user",
                    "content": f"Extract search patterns from this query: {user_input}",
                },
            ]

            pattern_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=pattern_messages,
                temperature=0.1,
                max_tokens=200,
            )

            potential_patterns = [
                p.strip()
                for p in pattern_response.choices[0].message.content.split("\n")
                if p.strip()
            ]

            # Filter to patterns with reasonable length and that we haven't explored yet
            search_patterns = [
                p
                for p in potential_patterns
                if len(p) > 2 and p not in explored_patterns
            ]

            if search_patterns:
                console.print(
                    f"[blue]Searching for patterns: {', '.join(search_patterns)}[/blue]"
                )

                for pattern in search_patterns:
                    explored_patterns.add(pattern)

                    # Escape the pattern for grep
                    escaped_pattern = (
                        pattern.replace("'", "'\\''")
                        .replace("(", "\\(")
                        .replace(")", "\\)")
                    )

                    try:
                        cmd = f"git grep -n '{escaped_pattern}'"
                        result = run_command(repo_path, cmd)

                        if result:
                            # Summarize grep results
                            summarized_results = summarize_grep_results(
                                result, pattern, model
                            )

                            # Add to context
                            context.append(
                                {
                                    "role": "user",
                                    "content": f"Search results for '{pattern}':\n```\n{summarized_results}\n```",
                                }
                            )
                    except Exception as e:
                        console.print(
                            f"[yellow]Warning: Error searching for '{pattern}': {e}[/yellow]"
                        )

        # Process the response
        with Progress(
            SpinnerColumn(), TextColumn("[bold blue]Thinking..."), console=console
        ) as progress:
            _task = progress.add_task("Processing", total=None)

            client = get_openai_client()
            response = client.chat.completions.create(
                model=model,
                messages=context,
                temperature=0.1,
            )

            answer = response.choices[0].message.content

        # Add the response to context for future questions
        context.append({"role": "assistant", "content": answer})

        # Display the answer
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(Panel(Markdown(answer), border_style="green"))

    return "Chat session ended"
