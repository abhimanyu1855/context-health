"""Typer CLI entry-point for Context Health."""

import typer

app = typer.Typer(
    name="context-health",
    help="Measure the health of an AI coding-agent context.",
    no_args_is_help=True,
)


@app.command()
def chat(
    model: str = typer.Option("claude-sonnet-4-20250514", help="Anthropic model to use."),
    max_tokens: int = typer.Option(1024, help="Max tokens per response."),
) -> None:
    """Start an interactive chat session with context-health tracking."""
    typer.echo(f"[context-health] chat session placeholder (model={model}, max_tokens={max_tokens})")


@app.command()
def version() -> None:
    """Print the current version."""
    from context_health import __version__

    typer.echo(f"context-health {__version__}")


if __name__ == "__main__":
    app()
