"""
Command line interface for the TLDR package.
"""

import weave
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from tldr.utils import find_git_root
from tldr.agent import run_agent
from tldr.chat import chat_loop

app = typer.Typer()
console = Console()


@weave.op()
@app.command()
def summarize(
    repo_path: Path = typer.Argument(
        Path.cwd(),
        help="Path to the GitHub repository",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    model: str = typer.Option(
        "gpt-4o",
        help="OpenAI model to use (e.g., gpt-4o, gpt-3.5-turbo)",
    ),
    max_context_tokens: int = typer.Option(
        8000, help="Approximate maximum number of tokens in the context window"
    ),
    chat: bool = typer.Option(
        False,
        "--chat",
        "-c",
        help="Start an interactive chat session after generating the summary",
    ),
    md: bool = typer.Option(
        False,
        "--md",
        help="Output the summary as plain markdown instead of rich text, for easy copying to GitHub",
    ),
    output_file: Path = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Save the summary to the specified file in markdown format",
    ),
):
    """
    Summarize the current GitHub PR in the specified repository.
    With --chat flag, starts an interactive chat session about the PR.
    With --md flag, outputs the summary as plain markdown for easy copying to GitHub.
    With --output-file flag, saves the summary to the specified file in markdown format.
    """
    console.print(Panel.fit("PR Summarizer", title="tldr", border_style="blue"))

    # Find git root
    git_repo_path = find_git_root(repo_path)
    console.print(f"Repository path: [bold]{git_repo_path}[/bold]")
    console.print(f"Using OpenAI model: [bold]{model}[/bold]")
    console.print(f"Target context size: [bold]{max_context_tokens}[/bold] tokens")

    # Generate the summary - use run_agent which handles the async event loop
    summary = run_agent(git_repo_path, model)

    # Save to output file if specified
    if output_file:
        output_file.write_text(summary)
        console.print(f"\n[bold green]Summary saved to:[/bold green] {output_file}")

    # Display the summary
    console.print("\n[bold green]PR Summary:[/bold green]")
    if md:
        # Output as plain markdown for copying to GitHub
        console.print("\n[bold blue]Markdown summary (ready to copy):[/bold blue]")
        print("\n" + summary + "\n")
    else:
        # Display with rich formatting
        console.print(Panel(Markdown(summary), border_style="green"))

    # If chat mode is enabled, start the chat session
    if chat:
        console.print("\n[bold blue]Starting chat mode...[/bold blue]")
        chat_loop(git_repo_path, model, summary)
