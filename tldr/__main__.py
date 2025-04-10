"""
Main entry point for tldr package when run as a module.
"""

import weave
from tldr.cli import app
from tldr.agent import publish_prompts


def main():
    """
    Entry point for the script when run directly
    """
    # Initialize weave with a standalone name
    weave.init("tldr")
    publish_prompts()
    # Run the app
    app()


if __name__ == "__main__":
    # This ensures the script runs as standalone
    main()
