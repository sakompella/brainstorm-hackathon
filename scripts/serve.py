"""
Web server for the Neural Data Viewer.

Legacy entrypoint that serves the frontend and bridges browser WebSocket
connections to the upstream data stream.
"""

import typer
from fastapi import FastAPI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scripts.backend import BrowserServer, SharedState
from scripts.backend import create_app as create_backend_app
from scripts.backend import main as run_backend
from scripts.static_assets import resolve_static_dir

VIEWER_DIR = resolve_static_dir()

cli = typer.Typer(help="Neural Data Viewer Web Server")
console = Console()


def create_app(
    state: SharedState | None = None,
    server: BrowserServer | None = None,
) -> FastAPI:
    """Create the legacy FastAPI application with same-origin /ws support."""
    app_state = state or SharedState()
    app_server = server or BrowserServer(app_state)
    return create_backend_app(app_state, app_server, VIEWER_DIR)


app = create_app()


@cli.command()
def main(
    upstream_url: str = typer.Option(
        "ws://localhost:8765",
        "--upstream-url",
        "-u",
        help="Upstream data stream WebSocket URL",
    ),
    port: int = typer.Option(8000, help="Web server port"),
    host: str = typer.Option("localhost", help="Host to bind to"),
) -> None:
    """
    Start the Neural Data Viewer web server.

    This legacy command serves static files and exposes /ws for the frontend,
    forwarding data from the upstream stream_data.py server.
    """
    # Print header
    console.print()
    console.print(
        Panel(
            "[bold cyan]Neural Data Viewer Server[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()

    # Print configuration
    config_table = Table(show_header=False, box=None, padding=(0, 1))
    config_table.add_row("[dim]Web Server[/dim]", f"[cyan]http://{host}:{port}[/cyan]")
    config_table.add_row(
        "[dim]Data Stream[/dim]", f"[cyan]{upstream_url}[/cyan] (upstream)"
    )
    config_table.add_row("[dim]Browser WS[/dim]", f"[cyan]ws://{host}:{port}/ws[/cyan]")

    console.print(config_table)
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[cyan]Web Viewer[/cyan]", f"http://{host}:{port}")

    console.print(
        Panel(
            table,
            title="[bold green]🚀 Server Running[/bold green]",
            border_style="green",
        )
    )
    console.print()
    console.print(
        "[dim]Note: Web clients connect to this server at /ws; this server connects upstream.[/dim]"
    )
    console.print()

    run_backend(
        upstream_url=upstream_url,
        host=host,
        port=port,
        static_dir=str(VIEWER_DIR),
        process=True,
        difficulty="hard",
        out_hz=20.0,
        stateless=True,
        log_level="INFO",
    )


if __name__ == "__main__":
    cli()
