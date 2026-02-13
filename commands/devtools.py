import typer
import subprocess
import json
from typing import Optional

app = typer.Typer()


def _run(cmd: list[str], timeout: int = 30) -> str:
    """Run a shell command and return its stdout. Stderr is included on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return f"Command failed (exit {result.returncode}):\n{result.stderr.strip()}"
        return result.stdout.strip()
    except FileNotFoundError:
        return f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"


# -----------------------------------------------------
# GitHub CLI wrappers (read-only)
# -----------------------------------------------------

@app.command()
def list_prs(
    state: str = typer.Option("open", "--state", help="Filter by state: open, closed, merged, all"),
    limit: int = typer.Option(10, "--limit", help="Max number of PRs to show"),
):
    """Lists pull requests in the current repo using gh CLI."""
    output = _run(["gh", "pr", "list", "--state", state, "--limit", str(limit)])
    typer.echo(output)
    return output


@app.command()
def view_pr(
    number: int = typer.Argument(..., help="PR number to view"),
):
    """Views details of a specific pull request."""
    output = _run(["gh", "pr", "view", str(number)])
    typer.echo(output)
    return output


@app.command()
def list_issues(
    state: str = typer.Option("open", "--state", help="Filter by state: open, closed, all"),
    limit: int = typer.Option(10, "--limit", help="Max number of issues to show"),
):
    """Lists issues in the current repo using gh CLI."""
    output = _run(["gh", "issue", "list", "--state", state, "--limit", str(limit)])
    typer.echo(output)
    return output


@app.command()
def view_issue(
    number: int = typer.Argument(..., help="Issue number to view"),
):
    """Views details of a specific issue."""
    output = _run(["gh", "issue", "view", str(number)])
    typer.echo(output)
    return output


@app.command()
def list_runs(
    limit: int = typer.Option(10, "--limit", help="Max number of workflow runs to show"),
):
    """Lists recent CI/CD workflow runs using gh CLI."""
    output = _run(["gh", "run", "list", "--limit", str(limit)])
    typer.echo(output)
    return output


@app.command()
def view_run(
    run_id: str = typer.Argument(..., help="Workflow run ID to view"),
):
    """Views details of a specific workflow run."""
    output = _run(["gh", "run", "view", run_id])
    typer.echo(output)
    return output


@app.command()
def repo_status():
    """Shows repo info (gh repo view) combined with git status."""
    repo_info = _run(["gh", "repo", "view"])
    git_status = _run(["git", "status", "--short"])
    git_branch = _run(["git", "branch", "--show-current"])

    output = f"--- Repo Info ---\n{repo_info}\n\n--- Branch ---\n{git_branch}\n\n--- Git Status ---\n{git_status or '(clean)'}"
    typer.echo(output)
    return output


# -----------------------------------------------------
# Claude CLI wrappers (read-only)
# -----------------------------------------------------

@app.command()
def claude_status():
    """Checks if Claude Code CLI is available and shows version."""
    version = _run(["claude", "--version"])
    output = f"Claude CLI: {version}"
    typer.echo(output)
    return output


# -----------------------------------------------------
# Cursor / Agent PR helpers
# -----------------------------------------------------

@app.command()
def list_agent_prs(
    limit: int = typer.Option(10, "--limit", help="Max PRs to show"),
    author: str = typer.Option("", "--author", help="Filter by PR author (e.g. 'app/cursor-ai' or bot username)"),
):
    """Lists PRs likely created by AI agents (Cursor, Claude, etc.) by filtering on author or branch naming."""
    if author:
        output = _run(["gh", "pr", "list", "--limit", str(limit), "--author", author, "--state", "all"])
    else:
        # Fallback: search for PRs with common agent branch prefixes
        output = _run(["gh", "pr", "list", "--limit", str(limit), "--state", "all", "--search", "head:claude/ OR head:cursor/ OR head:devin/ OR head:bot/"])
    typer.echo(output)
    return output


# -----------------------------------------------------
# Entry point
# -----------------------------------------------------
def main():
    app()


if __name__ == "__main__":
    main()
