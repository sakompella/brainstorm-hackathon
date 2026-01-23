"""
Web server for the Neural Data Viewer.

This server serves the static web files. Web clients connect directly
to the data stream (stream_data.py) instead of through this server.
"""

from pathlib import Path

import typer
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

VIEWER_DIR = Path(__file__).parent.parent / "example_app"

cli = typer.Typer(help="Neural Data Viewer Web Server")
console = Console()


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Neural Data Viewer")

    @app.get("/")
    async def index() -> FileResponse:
        """Serve index.html."""
        return FileResponse(VIEWER_DIR / "index.html")

    # Mount static files (must come after explicit routes)
    app.mount("/", StaticFiles(directory=VIEWER_DIR), name="static")

    return app


app = create_app()


@cli.command()
def main(
    port: int = typer.Option(8000, help="Web server port"),
    host: str = typer.Option("localhost", help="Host to bind to"),
) -> None:
    """
    Start the Neural Data Viewer web server.

    This server serves static files. Web clients connect directly
    to the data stream (ws://localhost:8765) instead of through this server.
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
        "[dim]Data Stream[/dim]", "[cyan]ws://localhost:8765[/cyan] (direct connection)"
    )

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
        "[dim]Note: Web clients connect directly to the data stream at ws://localhost:8765[/dim]"
    )
    console.print()

    uvicorn.run(
        create_app(),
        host=host,
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    cli()
