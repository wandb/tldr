"""
LLM Agent functionality for analyzing PR content.
"""

import weave
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from tldr.utils import run_command, get_openai_client, IGNORE_PATTERNS
from tldr.git import get_current_pr, process_pr_files, get_file_diffs

console = Console()


# Define prompt objects for reuse
CODE_ANALYZER_PROMPT = weave.MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are an expert code analyzer. Your task is to identify which sections of code are most "
                "relevant to understanding a specific change indicated by a diff. Extract only the most "
                "important sections needed to understand the context of the changes."
            ),
        },
        {
            "role": "user",
            "content": (
                "Given this diff that shows changes to a file:\n\n```\n{diff_summary}\n```\n\n"
                "And the full file content:\n\n```\n{file_content_preview}\n```\n\n"
                "Extract only the parts of the file that are crucial for understanding the context of the changes. "
                "Include:\n"
                "1. The changed sections plus surrounding context (functions/classes containing changes)\n"
                "2. Important imports or definitions referenced by the changes\n"
                "3. Any critical interfaces or data structures affected\n\n"
                "Exclude unrelated code. Respond with only the extracted code, preserving its structure."
            ),
        },
    ]
)

GREP_SUMMARIZER_PROMPT = weave.MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are an expert code analyzer. Your task is to analyze search results from a code search "
                "and extract only the most relevant and unique occurrences that help understand how a pattern "
                "is used in the codebase."
            ),
        },
        {
            "role": "user",
            "content": (
                "I searched for the pattern '{pattern}' in a codebase and got these results:\n\n"
                "```\n{grep_results_preview}\n```\n\n"
                "Extract only the most informative and distinct occurrences that show different ways "
                "this pattern is used. Group similar usages together and summarize repetitive patterns. "
                "Preserve file paths and line numbers. Aim to provide a comprehensive understanding of how "
                "this pattern is used throughout the codebase in the most concise way possible."
            ),
        },
    ]
)

CODE_PATTERN_EXTRACTOR_PROMPT = weave.MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are an expert code analyst specializing in identifying meaningful patterns in code diffs. "
                "Your task is to thoroughly analyze code diffs and extract specific, unique identifiers or patterns "
                "that represent the most important changes. Focus on function names, method names, class names, "
                "variable names, and unique code patterns that have been added or modified. "
                "Look for patterns that would help identify where and how these changes are used throughout the codebase."
            ),
        },
        {
            "role": "user",
            "content": (
                "Analyze these code diffs in detail and identify 5-10 specific search patterns that will help find "
                "related code across the repository. The patterns should be precise enough to identify relevant code but "
                "not so broad that they return too many unrelated matches.\n\n"
                "{diff_content}\n\n"
                "For each important pattern you identify, provide:\n"
                "1. The exact pattern to search for (a function name, variable name, class name, or specific code syntax)\n"
                "2. A brief explanation of why this pattern is significant based on the diff\n"
                "3. The file type/extension to search in (if applicable)\n\n"
                "Format each pattern as 'PATTERN: <exact_search_string>' with your explanation underneath."
            ),
        },
    ]
)

OUTPUT_SUMMARIZER_PROMPT = weave.MessagesPrompt(
    [
        {
            "role": "system",
            "content": "You are an expert at analyzing and summarizing command output. Extract the most important information concisely.",
        },
        {
            "role": "user",
            "content": "Summarize the following command output from `{command}`:\n\n```\n{output_preview}\n```\n\nFocus on the key information and patterns.",
        },
    ]
)

PR_SUMMARIZER_PROMPT = weave.MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that summarizes GitHub pull requests. "
                "Your goal is to provide a concise, informative summary that helps reviewers "
                "understand the changes, their purpose, and potential impacts. "
                "If you need more information, request specific commands to run."
            ),
        },
    ]
)


def publish_prompts():
    weave.publish(CODE_ANALYZER_PROMPT, name="code-analyzer-prompt")
    weave.publish(GREP_SUMMARIZER_PROMPT, name="grep-summarizer-prompt")
    weave.publish(CODE_PATTERN_EXTRACTOR_PROMPT, name="code-pattern-extractor-prompt")
    weave.publish(OUTPUT_SUMMARIZER_PROMPT, name="output-summarizer-prompt")
    weave.publish(PR_SUMMARIZER_PROMPT, name="pr-summarizer-prompt")


@weave.op()
def extract_relevant_code(file_content: str, diff: str, model: str) -> str:
    """Use LLM to extract only the most relevant parts of a file based on the diff"""
    try:
        client = get_openai_client()

        # Prepare a diff summary if it's too large
        diff_summary = diff
        if len(diff) > 2000:
            diff_summary = diff[:2000] + "... [truncated for brevity]"

        # Prepare file content summary if it's too large
        if len(file_content) > 5000:
            # First pass with a smaller context-optimized model to identify relevant sections
            file_content_preview = file_content[:10000] + (
                "..." if len(file_content) > 10000 else ""
            )

            messages = CODE_ANALYZER_PROMPT.format(
                diff_summary=diff_summary, file_content_preview=file_content_preview
            )

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
            )

            return response.choices[0].message.content

        return file_content
    except Exception as e:
        console.print(
            f"[yellow]Warning: Error extracting relevant code: {e}. Using truncated version.[/yellow]"
        )
        # Fallback to simple truncation
        return file_content[:5000] + ("..." if len(file_content) > 5000 else "")


@weave.op()
def summarize_grep_results(grep_results: str, pattern: str, model: str) -> str:
    """Summarize grep results to focus on the most relevant matches"""
    if len(grep_results) <= 3000:
        return grep_results

    try:
        client = get_openai_client()

        grep_results_preview = grep_results[:7000] + (
            "..." if len(grep_results) > 7000 else ""
        )

        messages = GREP_SUMMARIZER_PROMPT.format(
            pattern=pattern, grep_results_preview=grep_results_preview
        )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
        )

        return response.choices[0].message.content
    except Exception as e:
        console.print(
            f"[yellow]Warning: Error summarizing grep results: {e}. Using truncated version.[/yellow]"
        )
        # Fallback to simple truncation
        return grep_results[:3000] + ("..." if len(grep_results) > 3000 else "")


@weave.op()
def extract_code_patterns(
    repo_path: Path, diffs: Dict[str, str], model: str
) -> List[str]:
    """Use LLM to extract specific code patterns from diffs for searching"""
    # Create a comprehensive prompt for the LLM with detailed diff analysis
    diff_content = ""
    for file_path, diff in diffs.items():
        # Include a reasonable amount of diff context while keeping the overall size manageable
        max_diff_length = 2000  # Increased for better context
        diff_snippet = diff[:max_diff_length] + (
            "..." if len(diff) > max_diff_length else ""
        )
        diff_content += f"\nFile: {file_path}\n```\n{diff_snippet}\n```\n"

    try:
        client = get_openai_client()

        messages = CODE_PATTERN_EXTRACTOR_PROMPT.format(diff_content=diff_content)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        content = response.choices[0].message.content

        # Extract patterns from the LLM response
        patterns = []
        for line in content.split("\n"):
            if line.startswith("PATTERN:"):
                pattern = line.split("PATTERN:", 1)[1].strip()
                if pattern and len(pattern) > 2:  # Avoid very short patterns
                    # Escape the pattern for use in grep
                    pattern = (
                        pattern.replace("'", "'\\''")
                        .replace("(", "\\(")
                        .replace(")", "\\)")
                    )
                    patterns.append(pattern)

        # If no patterns were extracted or patterns are too general, use a fallback approach
        if not patterns:
            console.print(
                "[yellow]Warning: Couldn't extract specific patterns from diffs, using fallback patterns[/yellow]"
            )
            # Extract basic patterns from file names and common code elements
            for file_path in diffs.keys():
                file_name = Path(file_path).name
                name_without_ext = Path(file_name).stem
                if len(name_without_ext) > 3:
                    patterns.append(name_without_ext)

        return patterns
    except Exception as e:
        console.print(f"[red]Error extracting code patterns: {e}[/red]")
        return []


def generate_git_grep_commands(patterns: List[str]) -> List[str]:
    """Generate git grep commands for the extracted patterns"""
    commands = []

    for pattern in patterns:
        # Basic git grep command that respects .gitignore
        cmd = f"git grep -n '{pattern}'"
        commands.append(cmd)

        # Add a context grep if the pattern seems like a function or class
        if any(char in pattern for char in "(){}") or pattern[0].isupper():
            cmd_with_context = f"git grep -n -A 3 -B 3 '{pattern}'"
            commands.append(cmd_with_context)

    return commands


def gather_initial_context(
    repo_path: Path, model: str
) -> Tuple[List[Dict[str, str]], Optional[Dict[str, Any]]]:
    """Gather initial context about the PR"""
    # Start with the PR summarizer prompt
    context = PR_SUMMARIZER_PROMPT.format()

    # Get PR metadata
    pr_info = get_current_pr(repo_path)
    if not pr_info:
        console.print("[red]No active PR found in this repository[/red]")
        return context, None

    # Add PR title and description
    context.append(
        {
            "role": "user",
            "content": f"I need to summarize PR #{pr_info['number']}: {pr_info['title']}\n\nDescription: {pr_info['body']}",
        }
    )

    # Get diff stats
    context.append(
        {
            "role": "user",
            "content": f"PR Stats: {pr_info['additions']} additions, {pr_info['deletions']} deletions",
        }
    )

    # Process PR files
    files_info = process_pr_files(repo_path, pr_info)
    processed_files = files_info["processed_files"]
    ignored_files = files_info["ignored_files"]
    binary_files = files_info["binary_files"]

    # Add summary of processed files to context
    files_list = "\n".join(
        [
            f"- {f['path']} ({f['additions']} additions, {f['deletions']} deletions)"
            for f in processed_files
        ]
    )

    if ignored_files or binary_files:
        files_list += f"\n\nNote: {len(ignored_files)} files were ignored and {len(binary_files)} binary files were skipped."

    context.append({"role": "user", "content": f"Changed files:\n{files_list}"})

    # Get diffs for all files
    diffs = get_file_diffs(repo_path, pr_info, processed_files)

    # Get diffs and full contents of all modified files
    for file in processed_files:
        diff = diffs[file["path"]]

        # Add a compressed version of the diff to context
        console.print(f"[blue]Adding optimized diff for {file['path']}[/blue]")
        context.append(
            {
                "role": "user",
                "content": f"Diff for {file['path']}:\n```\n{diff[:2000]}{'...' if len(diff) > 2000 else ''}\n```",
            }
        )

        # Get the full file contents and add only relevant parts
        try:
            file_content = run_command(repo_path, f"cat {file['path']}")

            # Extract only relevant code portions
            console.print(f"[blue]Extracting relevant code from {file['path']}[/blue]")
            relevant_code = extract_relevant_code(file_content, diff, model)

            context.append(
                {
                    "role": "user",
                    "content": f"Relevant code from {file['path']}:\n```\n{relevant_code}\n```",
                }
            )
        except Exception as e:
            console.print(
                f"[yellow]Warning: Couldn't read {file['path']}: {e}[/yellow]"
            )

    # Extract patterns from diffs and generate git grep commands
    if diffs:
        console.print("[blue]Extracting code patterns from diffs using LLM...[/blue]")
        patterns = extract_code_patterns(repo_path, diffs, model)

        if patterns:
            console.print(f"[blue]Found {len(patterns)} search patterns:[/blue]")
            for pattern in patterns:
                console.print(f"  - {pattern}")

            git_grep_commands = generate_git_grep_commands(patterns)

            # Execute git grep commands and add summarized results to context
            context.append(
                {
                    "role": "user",
                    "content": "Code usage analysis results:",
                }
            )

            for cmd in git_grep_commands:
                try:
                    result = run_command(repo_path, cmd)
                    if result:
                        # Extract the pattern from the command
                        pattern = cmd.split("'")[1] if "'" in cmd else "pattern"

                        # Summarize grep results
                        console.print(
                            f"[blue]Summarizing grep results for pattern: {pattern}[/blue]"
                        )
                        summarized_results = summarize_grep_results(
                            result, pattern, model
                        )

                        context.append(
                            {
                                "role": "user",
                                "content": f"Results for `{cmd}`:\n```\n{summarized_results}\n```",
                            }
                        )
                except Exception:
                    pass  # Silently continue if a grep fails, as some patterns might not match

    # Calculate approximate token count for context
    total_chars = sum(len(msg["content"]) for msg in context)
    estimated_tokens = total_chars / 4  # Rough estimate: ~4 chars per token
    console.print(
        f"[blue]Estimated context size: ~{estimated_tokens:.0f} tokens[/blue]"
    )

    return context, pr_info


@weave.op()
def process_llm_response(
    repo_path: Path, context: List[Dict[str, str]], model: str, iterations: int
) -> Tuple[List[Dict[str, str]], int, Optional[str]]:
    """Process LLM response and handle command execution if needed"""
    client = get_openai_client()

    response = client.chat.completions.create(
        model=model,
        messages=context,
        temperature=0.1,
    )

    content = response.choices[0].message.content
    new_context = context.copy()
    new_iterations = iterations + 1

    # Check if the assistant is asking for more information
    if "COMMAND:" in content:
        # Extract the command
        command_parts = content.split("COMMAND:", 1)
        reasoning = command_parts[0].strip()
        command = command_parts[1].strip().split("\n")[0].strip()

        # Add the assistant's reasoning to the context
        new_context.append({"role": "assistant", "content": reasoning})

        # Run the command and get the output
        output = run_command(repo_path, command)

        # Truncate or summarize command output if it's very large
        if len(output) > 3000:
            console.print(
                f"[yellow]Command output is large ({len(output)} chars), summarizing...[/yellow]"
            )
            try:
                client = get_openai_client()
                output_preview = output[:7000] + ("..." if len(output) > 7000 else "")

                summarize_messages = OUTPUT_SUMMARIZER_PROMPT.format(
                    command=command, output_preview=output_preview
                )

                summary_response = client.chat.completions.create(
                    model=model,
                    messages=summarize_messages,
                    temperature=0.1,
                    max_tokens=1000,
                )
                output_summary = summary_response.choices[0].message.content
                output = f"[Summarized output]:\n{output_summary}\n\n[Original output was {len(output)} chars and was summarized]"
            except Exception as e:
                console.print(
                    f"[yellow]Failed to summarize output: {e}. Using truncated version.[/yellow]"
                )
                output = (
                    output[:3000]
                    + f"\n\n[Output truncated, total length: {len(output)} chars]"
                )

        # Add the command output to the context
        new_context.append(
            {
                "role": "user",
                "content": f"Command output for `{command}`:\n```\n{output}\n```",
            }
        )

        return new_context, new_iterations, None
    else:
        # The assistant has provided a final summary
        new_context.append({"role": "assistant", "content": content})
        return new_context, new_iterations, content


@weave.op()
def agent_loop(repo_path: Path, model: str) -> str:
    """Run the agent loop to summarize the PR"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[bold blue]Gathering initial context...", total=None)

        # Initial context gathering
        context, pr_info = gather_initial_context(repo_path, model)
        if not context or pr_info is None:
            return "Failed to gather context for PR summarization"

        progress.update(task, description="Processing with LLM...")

        # Agent loop
        max_iterations = 5
        iterations = 0
        final_summary = None

        while iterations < max_iterations and final_summary is None:
            progress.update(
                task,
                description=f"Processing with LLM (iteration {iterations + 1}/{max_iterations})...",
            )
            context, iterations, final_summary = process_llm_response(
                repo_path, context, model, iterations
            )

            if final_summary is None and iterations < max_iterations:
                progress.update(task, description="Running follow-up command...")

        if final_summary is None:
            return "Reached maximum number of iterations without a final summary."

        return final_summary
