"""
LLM Agent functionality for analyzing PR content.
"""

import weave
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from tldr.utils import run_command_async, get_openai_client
from tldr.git import get_current_pr, process_pr_files, PRDiff

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
                "You are an expert code analyst specializing in identifying critical dependency patterns in code diffs. "
                "Your task is to analyze code diffs and extract ONLY the most important identifiers that represent "
                "potential dependencies between the changed code and the rest of the codebase. "
                "Focus strictly on function names, method names, class names, and unique identifiers that are either:"
                "1. Modified or added in the diff AND likely used elsewhere in the codebase, or "
                "2. External dependencies that the changed code relies on or interacts with. "
                "Avoid common language keywords, generic patterns, or identifiers that would match too many irrelevant files. "
                "Your response MUST contain ONLY the 3-5 most critical patterns to search for, one pattern per line. "
                "Do NOT include any explanations or formatting."
            ),
        },
        {
            "role": "user",
            "content": (
                "Analyze these code diffs and identify ONLY 3-5 critical patterns that would help identify "
                "code dependencies or interactions between the changed code and the rest of the codebase:\\n\\n"
                "{diff_content}\\n\\n"
                "Return ONLY the 3-5 most important patterns, one pattern per line, focusing exclusively on "
                "function names, class names, or unique identifiers that might reveal dependency relationships."
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

# New prompt for evaluating context sufficiency
CONTEXT_EVALUATOR_PROMPT = weave.MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator for an AI assistant that summarizes GitHub pull requests. "
                "Your task is to determine if the provided context is sufficient to generate a comprehensive and accurate summary. "
                "The context includes PR metadata, diffs, relevant code snippets, and results from code searches (grep). "
                "If the context IS sufficient, respond ONLY with the word 'SUFFICIENT'. "
                "If the context IS NOT sufficient, explain CLEARLY what specific additional information is needed and provide ONE SINGLE command to retrieve it, formatted as 'COMMAND: <command_string>'. "
                "For example, 'COMMAND: git grep -n 'pattern_name'' or 'COMMAND: cat path/to/specific/file.py'. "
                "Be precise about the command needed. Do not ask general questions."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here is the current context gathered about the pull request:\\n\\n"
                "--- BEGIN CONTEXT --- {context_summary} --- END CONTEXT ---\\n\\n"
                "Is this context sufficient to write a high-quality PR summary? "
                "If yes, respond ONLY with 'SUFFICIENT'. "
                "If no, explain what's missing and provide the ONE specific command (e.g., git grep, cat) needed to get it, formatted as 'COMMAND: <command_string>'."
            ),
        },
    ]
)

# New prompt for selecting relevant files and line ranges
FILE_SELECTOR_PROMPT = weave.MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are an expert code dependency analyzer. Your task is to analyze pull request diffs and grep results "
                "to identify ONLY the most critical files that directly interact with or depend on the changed code. "
                "Focus exclusively on files where there is strong evidence of a dependency relationship with the modified code. "
                "Ignore files that merely contain similar patterns but don't show clear dependency evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                "Analyze the following pull request diffs:\\n\\n"
                "--- BEGIN DIFFS --- {combined_diff_content} --- END DIFFS ---\\n\\n"
                "And the following grep results from searching for patterns related to these diffs:\\n\\n"
                "--- BEGIN GREP RESULTS --- {combined_grep_output} --- END GREP RESULTS ---\\n\\n"
                "Identify ONLY the 2-3 most critical files that likely have a direct dependency relationship with the changed code. "
                "Look for evidence such as:\n"
                "1. Files that import, extend, or instantiate classes/functions modified in the diff\n"
                "2. Files where the changed functionality is clearly being called or referenced\n"
                "3. Parent classes or interfaces that the modified code implements\n\n"
                "For each selected file, provide a specific line range to examine (e.g., the function or class that shows the dependency).\n"
                "Return ONLY the identified selections in the format: 'file_path:start_line-end_line' (one per line).\n"
                "Do NOT include any other text, explanations, or files without clear dependency evidence."
            ),
        },
    ]
)

# New prompt for generating the final summary
FINAL_SUMMARIZER_PROMPT = weave.MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that summarizes GitHub pull requests. "
                "You have been provided with comprehensive context including PR metadata, diffs, relevant code snippets, and code search results. "
                "Your goal is to synthesize this information into a concise, informative summary that helps reviewers "
                "understand the changes, their purpose, and potential impacts. Focus on clarity and accuracy. Do not ask for more information."
            ),
        },
        {
            "role": "user",
            "content": (
                "Based on the following complete context, please generate a final summary for the pull request:\\n\\n"
                "--- BEGIN CONTEXT --- {context_summary} --- END CONTEXT ---\\n\\n"
                "Generate the summary now."
            ),
        },
    ]
)

# New prompt for summarizing pattern context within a file
FILE_PATTERN_CONTEXT_SUMMARIZER_PROMPT = weave.MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are an expert code analyzer. Your task is to summarize how a specific code pattern is used "
                "within a given file, based on provided match locations. Focus on the context surrounding the matches "
                "to explain their purpose and relevance."
            ),
        },
        {
            "role": "user",
            "content": (
                "The pattern '{pattern}' was found in the file '{file_path}' at the following lines:\\n"
                "{match_lines_summary}\\n\\n"
                "Here is the relevant content of the file '{file_path}':\\n\\n"
                "```\\n{file_content_preview}\\n```\\n\\n"
                "Summarize how the pattern '{pattern}' is used in this file, based on the context around the matched lines. "
                "Highlight the key functions, classes, or logic involved near the matches. "
                "Provide a concise explanation focusing on the pattern's role in this specific file."
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
    weave.publish(CONTEXT_EVALUATOR_PROMPT, name="context-evaluator-prompt")
    weave.publish(FINAL_SUMMARIZER_PROMPT, name="final-summarizer-prompt")
    weave.publish(
        FILE_PATTERN_CONTEXT_SUMMARIZER_PROMPT,
        name="file-pattern-context-summarizer-prompt",
    )
    weave.publish(FILE_SELECTOR_PROMPT, name="file-selector-prompt")


@weave.op()
async def extract_relevant_code(file_content: str, diff: str, model: str) -> str:
    """Use LLM to extract only the most relevant parts of a file based on the diff"""
    try:
        client = get_openai_client()

        # Prepare a diff summary if it's too large
        diff_summary = diff
        if len(diff) > 2000:
            diff_summary = diff[:2000] + "... [truncated for brevity]"

        # First pass with a smaller context-optimized model to identify relevant sections
        file_content_preview = file_content[:10000] + (
            "..." if len(file_content) > 10000 else ""
        )

        messages = CODE_ANALYZER_PROMPT.format(
            diff_summary=diff_summary, file_content_preview=file_content_preview
        )

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
        )

        return response.choices[0].message.content

    except Exception as e:
        console.print(
            f"[yellow]Warning: Error extracting relevant code: {e}. Using truncated version.[/yellow]"
        )
        # Fallback to simple truncation
        return file_content[:5000] + ("..." if len(file_content) > 5000 else "")


@weave.op()
async def summarize_grep_results(grep_results: str, pattern: str, model: str) -> str:
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

        response = await client.chat.completions.create(
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
async def extract_code_patterns(diff_content: str, model: str) -> List[str]:
    """Use LLM to extract specific code patterns from a combined diff string for searching,
    focusing on critical dependencies."""
    patterns = []
    if not diff_content.strip():  # Handle empty input diffs
        return patterns

    try:
        # Limit diff content size sent to LLM if necessary
        max_llm_diff_length = 100000  # Example limit, adjust as needed
        llm_diff_input = diff_content
        if len(diff_content) > max_llm_diff_length:
            llm_diff_input = (
                diff_content[:max_llm_diff_length] + "\n... [diff truncated] ..."
            )
            console.print(
                f"[yellow]Warning: Diff content truncated for pattern extraction LLM ({len(diff_content)} -> {max_llm_diff_length} chars)[/yellow]"
            )

        # Call the internal LLM function
        llm_response_content = await _call_pattern_extractor_llm(llm_diff_input, model)

        if llm_response_content:
            # Split the response by newline and process each line
            potential_patterns = llm_response_content.strip().split("\n")
            for pattern in potential_patterns:
                pattern = pattern.strip()
                if (
                    pattern and len(pattern) > 2
                ):  # Check if pattern is not empty and reasonably long
                    # Escape the pattern for use in grep
                    escaped_pattern = (
                        pattern.replace("'", "'\\''")
                        .replace("(", "\\(")
                        .replace(")", "\\)")
                    )
                    patterns.append(escaped_pattern)
                    console.print(
                        f"[blue]Extracted dependency pattern: {pattern}[/blue]"
                    )

        # Limit the number of patterns
        if len(patterns) > 5:
            console.print(
                f"[yellow]Limiting from {len(patterns)} to 5 most specific patterns[/yellow]"
            )
            # Sort by length (longer patterns are typically more specific)
            patterns = sorted(patterns, key=len, reverse=True)[:5]

        return patterns

    except Exception as e:
        console.print(f"[red]Error in extract_code_patterns: {e}[/red]")
        return []


async def _call_pattern_extractor_llm(diff_content: str, model: str) -> Optional[str]:
    """Internal function to call the LLM for pattern extraction."""
    try:
        client = get_openai_client()
        messages = CODE_PATTERN_EXTRACTOR_PROMPT.format(diff_content=diff_content)
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        console.print(f"[red]LLM call failed in _call_pattern_extractor_llm: {e}[/red]")
        return None


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


@weave.op()
async def evaluate_context_sufficiency(
    context: List[Dict[str, str]], model: str
) -> Dict[str, Any]:
    """
    Uses an LLM to evaluate if the gathered context is sufficient for PR summarization.

    Returns:
        Dict containing:
        - 'sufficient': bool - True if context is sufficient, False otherwise.
        - 'command': Optional[str] - The command to run if more context is needed.
        - 'reasoning': Optional[str] - The LLM's reasoning if more context is needed.
    """
    try:
        client = get_openai_client()

        # Create a condensed summary of the context for the prompt
        # TODO: Implement a more sophisticated context summarization if needed
        context_summary = "\n".join(
            [f"{msg['role']}: {msg['content'][:200]}..." for msg in context]
        )
        if len(context_summary) > 4000:
            context_summary = context_summary[:4000] + "... [truncated]"

        messages = CONTEXT_EVALUATOR_PROMPT.format(context_summary=context_summary)

        response = await client.chat.completions.create(
            model=model,  # Consider using a faster/cheaper model if appropriate
            messages=messages,
            temperature=0.1,
            max_tokens=300,
        )
        content = response.choices[0].message.content.strip()

        if content == "SUFFICIENT":
            return {"sufficient": True, "command": None, "reasoning": None}
        elif "COMMAND:" in content:
            parts = content.split("COMMAND:", 1)
            reasoning = parts[0].strip()
            command = parts[1].strip().split("\n")[0].strip()
            return {"sufficient": False, "command": command, "reasoning": reasoning}
        else:
            # Fallback: Assume insufficient if response is unexpected
            console.print(
                f"[yellow]Warning: Unexpected response from context evaluator: {content}[/yellow]"
            )
            # Maybe try a generic grep or ask user? For now, assume insufficient without a command.
            return {
                "sufficient": False,
                "command": None,
                "reasoning": "Context evaluator returned an unexpected response.",
            }

    except Exception as e:
        console.print(f"[red]Error evaluating context sufficiency: {e}[/red]")
        # Fallback: Assume insufficient on error
        return {
            "sufficient": False,
            "command": None,
            "reasoning": f"Error during context evaluation: {e}",
        }


@weave.op()
async def summarize_file_context_for_pattern(
    file_path: str,
    pattern: str,
    matches: List[Dict[str, Any]],  # e.g., [{'line': 123, 'content': '...'}, ...]
    file_content: str,
    model: str,
) -> str:
    """Summarize the context of a pattern's usage within a single file using LLM."""
    if not matches:
        return f"No matches provided for pattern '{pattern}' in file '{file_path}'."

    try:
        client = get_openai_client()

        # Create a summary of match lines
        match_lines_summary = "\n".join(
            [
                f"- Line {m['line']}: {m['content'][:100]}..." for m in matches[:10]
            ]  # Show first 10 matches max
        )
        if len(matches) > 10:
            match_lines_summary += f"\n... and {len(matches) - 10} more matches."

        # Preview file content if too large
        file_content_preview = file_content
        max_content_length = 8000  # Limit context sent to LLM
        if len(file_content) > max_content_length:
            # Try to center the preview around the first match
            first_match_line = matches[0]["line"]
            # Estimate character position (very rough)
            avg_line_len = (
                len(file_content) / file_content.count("\n", 0, len(file_content) - 1)
                if file_content.count("\n", 0, len(file_content) - 1) > 0
                else 80
            )
            estimated_pos = int(first_match_line * avg_line_len)
            start_pos = max(0, estimated_pos - max_content_length // 2)
            end_pos = min(len(file_content), start_pos + max_content_length)
            file_content_preview = file_content[start_pos:end_pos]
            if start_pos > 0:
                file_content_preview = (
                    "... [content truncated] ...\n" + file_content_preview
                )
            if end_pos < len(file_content):
                file_content_preview += "\n... [content truncated] ..."
            console.print(
                f"[yellow]Previewing content for {file_path} ({len(file_content)} -> {len(file_content_preview)} chars)[/yellow]"
            )

        messages = FILE_PATTERN_CONTEXT_SUMMARIZER_PROMPT.format(
            pattern=pattern,
            file_path=file_path,
            match_lines_summary=match_lines_summary,
            file_content_preview=file_content_preview,
        )

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,  # Allow reasonable space for summary
        )

        summary = response.choices[0].message.content.strip()
        return f"Usage summary for pattern '{pattern}' in '{file_path}':\n{summary}"

    except Exception as e:
        console.print(
            f"[red]Error summarizing pattern context for {file_path}: {e}[/red]"
        )
        # Fallback summary
        match_lines_summary = "\n".join([f"- Line {m['line']}" for m in matches[:5]])
        if len(matches) > 5:
            match_lines_summary += f"\n... ({len(matches)} total matches)."
        return f"Error summarizing pattern '{pattern}' usage in '{file_path}'. Matches found near lines:\n{match_lines_summary}\nError: {e}"


@weave.op()
async def select_relevant_files_and_lines(
    combined_diff_content: str,
    combined_grep_output: str,
    model: str,
) -> List[Dict[str, Any]]:
    """Use LLM to select only the most relevant files that show dependency relationships with the changed code."""
    if not combined_grep_output.strip():
        console.print("[yellow]No grep results provided to select files from.[/yellow]")
        return []

    try:
        client = get_openai_client()

        # Limit input size if necessary
        max_diff_len = 4000
        max_grep_len = 8000
        diff_preview = combined_diff_content
        grep_preview = combined_grep_output

        if len(diff_preview) > max_diff_len:
            diff_preview = diff_preview[:max_diff_len] + "... [diff truncated] ..."
        if len(grep_preview) > max_grep_len:
            grep_preview = (
                grep_preview[:max_grep_len] + "... [grep results truncated] ..."
            )

        console.print(
            "[blue]Analyzing diffs and grep results to identify dependency relationships...[/blue]"
        )

        messages = FILE_SELECTOR_PROMPT.format(
            combined_diff_content=diff_preview,
            combined_grep_output=grep_preview,
        )

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
        )

        message = response.choices[0].message
        if message.content is None:
            finish_reason = response.choices[0].finish_reason
            console.print(
                f"[yellow]Warning: LLM response content was None (finish_reason: {finish_reason}). Cannot select files.[/yellow]"
            )
            return []

        content = message.content.strip()
        console.print(f"[blue]Dependency analysis selected: {content}[/blue]")

        # Parse the file paths and line ranges
        selected_files = []
        line_pattern = re.compile(r"^(.*?):(\d+)-(\d+)$")

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue

            match = line_pattern.match(line)
            if match:
                file_path, start_str, end_str = match.groups()
                file_path = file_path.strip()

                if not file_path:
                    continue

                try:
                    start_line = int(start_str)
                    end_line = int(end_str)

                    if start_line > 0 and end_line >= start_line:
                        selected_files.append(
                            {
                                "file_path": file_path,
                                "start_line": start_line,
                                "end_line": end_line,
                            }
                        )
                    else:
                        console.print(
                            f"[yellow]Warning: Skipping invalid line range in selection: {line}[/yellow]"
                        )
                except ValueError:
                    console.print(
                        f"[yellow]Warning: Skipping item with non-integer line numbers: {line}[/yellow]"
                    )
            else:
                console.print(
                    f"[yellow]Warning: Skipping line that does not match expected format 'file:start-end': {line}[/yellow]"
                )

        console.print(
            f"[green]Selected {len(selected_files)} dependency-related files for review[/green]"
        )
        return selected_files

    except Exception:
        console.print("[red]Error selecting relevant files and lines:[/red]")
        console.print_exception()
        return []


def parse_grep_output(output: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parses output from 'git grep -n' into a dictionary.

    Args:
        output: The raw string output from 'git grep -n'.

    Returns:
        A dictionary where keys are file paths and values are lists of
        dictionaries, each containing 'line' (int) and 'content' (str).
    """
    matches_by_file = {}
    # Regex to capture file path, line number, and content
    # Handles potential colons in file paths on Windows by being less greedy
    # Assumes line numbers don't have colons.
    pattern = re.compile(r"^([^:]+):(\d+):(.*)$")
    for line in output.strip().split("\n"):
        match = pattern.match(line)
        if match:
            file_path, line_num_str, content = match.groups()
            line_num = int(line_num_str)
            if file_path not in matches_by_file:
                matches_by_file[file_path] = []
            matches_by_file[file_path].append(
                {"line": line_num, "content": content.strip()}
            )
    return matches_by_file


@weave.op()
async def generate_pr_summary(context: List[Dict[str, str]], model: str) -> str:
    """
    Generates the final PR summary using the provided context.
    Assumes the context is sufficient.
    """
    try:
        client = get_openai_client()

        # TODO: Implement context summarization/filtering if full context exceeds limit
        context_summary = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in context]
        )
        # Simple truncation for now
        if len(context_summary) > 15000:  # Adjust based on target model context window
            context_summary = context_summary[:15000] + "... [context truncated]"

        messages = FINAL_SUMMARIZER_PROMPT.format(context_summary=context_summary)

        response = await client.chat.completions.create(
            model=model,  # Use the main summarization model
            messages=messages,
            temperature=0.1,
            max_tokens=1500,  # Allow ample space for the summary
        )
        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        console.print(f"[red]Error generating final PR summary: {e}[/red]")
        return f"Error: Could not generate PR summary due to an internal error ({e})."


@weave.op()
async def generate_pr_summary_from_data(
    title: str,
    number: int,
    body: str,
    link: str,
    base_branch: str,
    head_branch: str,
    diffs: List[PRDiff],
    files_info: Dict[str, List[Dict]],
    repo_path: Path,
    model: str,
) -> str:
    """
    Generates a PR summary using provided PR data, including context refinement loop.
    """
    # Initialize context
    context = PR_SUMMARIZER_PROMPT.format()

    # Add PR title and description
    context.append(
        {
            "role": "user",
            "content": f"I need to summarize PR #{number}: {title}\n\nDescription: {body}\nLink: {link}\nBase branch: {base_branch}\nHead branch: {head_branch}",
        }
    )

    # Add summary of processed files to context
    processed_files = files_info["processed_files"]
    ignored_files = files_info["ignored_files"]
    binary_files = files_info["binary_files"]
    files_list = "\n".join(
        [
            f"- {f['path']} ({f['additions']} additions, {f['deletions']} deletions)"
            for f in processed_files
        ]
    )

    if ignored_files or binary_files:
        files_list += f"\n\nNote: {len(ignored_files)} files were ignored and {len(binary_files)} binary files were skipped."

    context.append({"role": "user", "content": f"Changed files:\n{files_list}"})

    # --- Process diffs and add relevant code from CHANGED files ---
    console.print("[blue]Analyzing changes in modified files...[/blue]")

    combined_diff_content = ""
    # List to hold coroutines for gathering relevant code in parallel
    relevance_coroutines = []
    relevance_file_paths = []

    for diff_item in diffs:
        file_path = diff_item["path"]
        diff = diff_item["content"]
        if not diff:
            continue

        combined_diff_content += f"--- Diff for {file_path} ---\n"
        combined_diff_content += diff
        combined_diff_content += "\n\n"

        # Add diff snippet immediately to context
        context.append(
            {
                "role": "user",
                "content": f"Diff snippet for {file_path}:\n```\n{diff[:500]}{'...' if len(diff) > 500 else ''}\n```",
            }
        )

        # Prepare coroutines for extracting relevant code
        try:
            # Ensure quoting for paths with spaces - use async command running
            file_content = await run_command_async(repo_path, f"cat '{file_path}'")
            console.print(
                f"[blue]Preparing to extract relevant code from {file_path}[/blue]"
            )

            # Schedule the coroutine to be run concurrently
            relevance_coroutines.append(
                extract_relevant_code(file_content, diff, model)
            )
            relevance_file_paths.append(file_path)

        except Exception as e:
            console.print(
                f"[yellow]Warning: Couldn't read {file_path} to extract relevant code: {e}[/yellow]"
            )

    # Process extract_code_patterns in parallel with the relevance tasks
    pattern_coroutine = None
    if combined_diff_content:
        console.print(
            "[blue]Extracting critical dependency patterns from diffs using LLM...[/blue]"
        )
        pattern_coroutine = extract_code_patterns(
            diff_content=combined_diff_content, model=model
        )

    # Gather all relevance results using asyncio.gather
    if relevance_coroutines:
        console.print(
            f"[blue]Awaiting {len(relevance_coroutines)} code relevance tasks...[/blue]"
        )
        try:
            relevance_results = await asyncio.gather(
                *relevance_coroutines, return_exceptions=True
            )

            for i, result in enumerate(relevance_results):
                file_path = relevance_file_paths[i]
                if isinstance(result, Exception):
                    console.print(
                        f"[yellow]Error getting relevant code for {file_path}: {result}[/yellow]"
                    )
                    continue

                relevant_code = result
                context.append(
                    {
                        "role": "user",
                        "content": f"Relevant code from {file_path}:\n```\n{relevant_code}\n```",
                    }
                )
                console.print(f"[green]Added relevant code from {file_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error gathering relevance results: {e}[/red]")
    # --- End processing diffs ---

    # --- Extract patterns and select relevant dependency-related context ---
    all_grep_output = ""
    if combined_diff_content:
        patterns = []
        if pattern_coroutine:
            try:
                patterns = await pattern_coroutine
            except Exception as e:
                console.print(f"[red]Error extracting code patterns: {e}[/red]")

        if patterns:
            console.print(
                f"[blue]Found {len(patterns)} dependency-related patterns to investigate[/blue]"
            )
            grep_patterns = [p for p in patterns if len(p.strip()) > 3]

            if grep_patterns:
                console.print(
                    f"[blue]Searching codebase for dependency relationships ({len(grep_patterns)} patterns)...[/blue]"
                )

                # Run grep searches in parallel using asyncio directly with run_command_async
                grep_tasks = []
                for pattern in grep_patterns:
                    cmd = f"git grep -n '{pattern}'"
                    grep_tasks.append((pattern, cmd))

                # Using asyncio.gather to run grep commands (non-async) concurrently
                # We'll use a helper function to handle the subprocess calls
                grep_results = await asyncio.gather(
                    *[run_command_async(repo_path, cmd) for _, cmd in grep_tasks],
                    return_exceptions=True,
                )

                for i, result in enumerate(grep_results):
                    pattern = grep_tasks[i][0]
                    if isinstance(result, Exception):
                        console.print(
                            f"[yellow]Error searching for pattern '{pattern}': {result}[/yellow]"
                        )
                        continue

                    grep_output = result
                    if grep_output:
                        filtered_output = ""
                        # Filter out the changed files from grep results to focus on dependencies
                        changed_file_paths = [d["path"] for d in diffs]
                        for line in grep_output.strip().splitlines():
                            file_in_line = line.split(":", 1)[0] if ":" in line else ""
                            if file_in_line and file_in_line not in changed_file_paths:
                                filtered_output += line + "\n"

                        if filtered_output:
                            all_grep_output += (
                                f"--- Dependency results for pattern: {pattern} ---\n"
                            )
                            all_grep_output += filtered_output.strip() + "\n\n"
                            console.print(
                                f"[green]Found potential dependencies for pattern '{pattern}'[/green]"
                            )

                # Now, use the LLM to select specific dependency-related files based on grep output
                if all_grep_output.strip():
                    console.print(
                        "[blue]Analyzing dependency relationships from search results...[/blue]"
                    )
                    selected_sections = await select_relevant_files_and_lines(
                        combined_diff_content=combined_diff_content,
                        combined_grep_output=all_grep_output,
                        model=model,
                    )

                    if selected_sections:
                        console.print(
                            f"[blue]Found {len(selected_sections)} dependency relationships to include in analysis[/blue]"
                        )
                        context.append(
                            {
                                "role": "user",
                                "content": "Critical dependency relationships identified:",
                            }
                        )

                        # Get all snippets concurrently
                        snippet_tasks = []
                        for section in selected_sections[
                            :3
                        ]:  # Limit to 3 most relevant dependencies
                            try:
                                file_path = section["file_path"]
                                start_line = section["start_line"]
                                end_line = section["end_line"]

                                # Prepare the command but execute them concurrently later
                                escaped_file_path = file_path.replace("'", "'\\''")
                                sed_cmd = f"sed -n '{start_line},{end_line}p' '{escaped_file_path}'"
                                snippet_tasks.append(
                                    (file_path, start_line, end_line, sed_cmd)
                                )
                            except KeyError as e:
                                console.print(
                                    f"[yellow]Missing key in dependency section: {e}[/yellow]"
                                )

                        # Execute all sed commands concurrently with run_command_async
                        if snippet_tasks:
                            snippet_results = await asyncio.gather(
                                *[
                                    run_command_async(repo_path, cmd)
                                    for _, _, _, cmd in snippet_tasks
                                ],
                                return_exceptions=True,
                            )

                            for i, result in enumerate(snippet_results):
                                file_path, start_line, end_line, _ = snippet_tasks[i]
                                if isinstance(result, Exception):
                                    console.print(
                                        f"[yellow]Error extracting dependency context from {file_path}: {result}[/yellow]"
                                    )
                                    continue

                                snippet = result
                                if snippet:
                                    context.append(
                                        {
                                            "role": "user",
                                            "content": f"Dependency in {file_path} (Lines {start_line}-{end_line}):\n```\n{snippet.strip()}\n```",
                                        }
                                    )
                    else:
                        console.print(
                            "[yellow]No significant dependency relationships found[/yellow]"
                        )
                else:
                    console.print(
                        "[yellow]No dependency references found in the codebase[/yellow]"
                    )
            else:
                console.print(
                    "[yellow]No viable dependency patterns identified for search[/yellow]"
                )
    # --- End dependency analysis ---

    # Calculate approximate token count for context
    total_chars = sum(len(msg["content"]) for msg in context)
    estimated_tokens = total_chars / 4
    console.print(
        f"[blue]Estimated context size: ~{estimated_tokens:.0f} tokens[/blue]"
    )

    # --- Start context refinement loop ---
    max_iterations = 5
    iterations = 0
    final_summary = None

    while iterations < max_iterations:
        iterations += 1
        console.print(
            f"[cyan]Evaluating context (iteration {iterations}/{max_iterations})...[/cyan]"
        )

        evaluation_result = await evaluate_context_sufficiency(context, model)

        if evaluation_result["sufficient"]:
            console.print(
                "[green]Context sufficient. Generating final summary.[/green]"
            )
            final_summary = await generate_pr_summary(context, model)
            break
        elif evaluation_result["command"]:
            command = evaluation_result["command"]
            reasoning = evaluation_result["reasoning"]
            console.print(
                f"[yellow]Context insufficient. Need to run: {command}[/yellow]"
            )
            if reasoning:
                console.print(f"[yellow]Reason: {reasoning}[/yellow]")
                context.append({"role": "assistant", "content": reasoning})

            try:
                console.print(f"[blue]Running command: {command[:100]}...[/blue]")
                output = await run_command_async(repo_path, command)
                output_summary = output
                if len(output) > 3000:
                    console.print(
                        f"[yellow]Command output is large ({len(output)} chars), summarizing...[/yellow]"
                    )
                    try:
                        client = get_openai_client()
                        output_preview = output[:7000] + (
                            "..." if len(output) > 7000 else ""
                        )
                        summarize_messages = OUTPUT_SUMMARIZER_PROMPT.format(
                            command=command, output_preview=output_preview
                        )
                        summary_response = await client.chat.completions.create(
                            model=model,
                            messages=summarize_messages,
                            temperature=0.1,
                            max_tokens=1000,
                        )
                        output_summary = summary_response.choices[0].message.content
                        output_summary = f"[Summarized output]:\n{output_summary}\n\n[Original output was {len(output)} chars and was summarized]"
                    except Exception as e:
                        console.print(
                            f"[yellow]Failed to summarize output: {e}. Using truncated version.[/yellow]"
                        )
                        output_summary = (
                            output[:3000]
                            + f"\n\n[Output truncated, total length: {len(output)} chars]"
                        )

                context.append(
                    {
                        "role": "user",
                        "content": f"Command output for `{command}`:\n```\n{output_summary}\n```",
                    }
                )

            except Exception as e:
                console.print(f"[red]Error running command '{command}': {e}[/red]")
                context.append(
                    {
                        "role": "user",
                        "content": f"Error executing command `{command}`: {e}",
                    }
                )
                # Consider stopping if a command fails? For now, let it continue.
        else:
            reason = evaluation_result.get("reasoning", "Unknown reason.")
            console.print(
                f"[red]Context deemed insufficient, but no command provided. Reason: {reason}[/red]"
            )
            final_summary = (
                f"Error: Could not gather sufficient context. Last reason: {reason}"
            )
            break

    if final_summary is None:  # Handles reaching max iterations
        if iterations == max_iterations:
            console.print(
                "[yellow]Reached maximum iterations. Attempting final summary with current context.[/yellow]"
            )
            final_summary = await generate_pr_summary(context, model)
            if not final_summary.startswith("Error:"):
                final_summary = (
                    "[Warning: Max iterations reached, summary may be incomplete]\n\n"
                    + final_summary
                )
            else:  # If generate_pr_summary also errored
                final_summary = "Error: Reached maximum iterations and failed to generate final summary."
        else:  # Should not happen if loop exits cleanly, but as a safeguard
            final_summary = (
                "Error: Loop finished unexpectedly without generating a summary."
            )

    return final_summary


@weave.op()
async def agent_loop(repo_path: Path, model: str) -> str:
    """
    Fetches PR data and calls the op to generate the summary.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[bold blue]Fetching PR data...", total=None)

        pr_info = get_current_pr(repo_path)
        if not pr_info:
            progress.stop()
            console.print("[red]No active PR found in this repository[/red]")
            return "Error: No active PR found."

        # Debug information about the PR
        console.print(f"[blue]PR #{pr_info['number']}: {pr_info['title']}[/blue]")
        console.print(
            f"[blue]Branches: {pr_info['head_branch']} → {pr_info['base_branch']}[/blue]"
        )
        console.print(f"[blue]Number of diffs: {len(pr_info['diffs'])}[/blue]")
        for diff in pr_info["diffs"][:3]:  # Show first 3 diffs for debugging
            console.print(f"[blue]Diff file: {diff['path']}[/blue]")

        progress.update(task, description=f"Processing PR #{pr_info['number']}...")

        files_info = process_pr_files(repo_path, pr_info)
        processed_files = files_info["processed_files"]
        console.print(f"[blue]Number of processed files: {len(processed_files)}[/blue]")

        if not processed_files:
            progress.stop()
            console.print("[yellow]No processable files found in the PR.[/yellow]")
            # Optionally generate a basic summary from metadata
            return f"Summary for PR #{pr_info['number']}: {pr_info['title']}\n\nDescription: {pr_info['body']}\n\nNo code changes detected or all files were ignored/binary."

        progress.update(task, description="Generating PR summary...")
        summary = await generate_pr_summary_from_data(
            title=pr_info["title"],
            number=pr_info["number"],
            body=pr_info["body"],
            link=pr_info["link"],
            base_branch=pr_info["base_branch"],
            head_branch=pr_info["head_branch"],
            diffs=pr_info["diffs"],
            files_info=files_info,
            repo_path=repo_path,
            model=model,
        )

        progress.update(task, description="Summary generated.", completed=True, total=1)

    return summary


# Add an entry point function for running the event loop
def run_agent(repo_path: Path, model: str) -> str:
    """
    Entry point function that runs the asyncio event loop.
    This is needed since weave.op() functions might not handle async functions properly.
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(agent_loop(repo_path, model))
