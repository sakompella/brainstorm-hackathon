"""End-to-end Playwright tests for the Neural Data Viewer.

Requires a running backend at http://localhost:8000 (e.g. via Docker):
    docker run -p 8000:8000 -v $(pwd)/data:/app/data brainstorm

Run with:
    uv run pytest tests/test_e2e.py --base-url http://localhost:8000
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

# All tests in this module need a live server
pytestmark = pytest.mark.e2e

BASE_WAIT = 3_000  # ms — time for WebSocket to connect + first data


@pytest.fixture(autouse=True)
def _navigate(page: Page, base_url: str) -> None:
    """Navigate to the app and wait for WebSocket to connect."""
    page.goto(base_url)
    # Wait until status text changes from "Disconnected" to "Connected"
    page.locator("#status-text").filter(has_text="Connected").wait_for(timeout=10_000)


class TestPageLoad:
    """Verify the page structure loads correctly."""

    def test_title(self, page: Page) -> None:
        expect(page).to_have_title("Neural Data Viewer")

    def test_heading(self, page: Page) -> None:
        expect(page.locator("h1").first).to_have_text("Neural Activity")

    def test_status_bar_elements_present(self, page: Page) -> None:
        expect(page.locator("#status-text")).to_be_visible()
        expect(page.locator("#time-display")).to_be_visible()
        expect(page.locator("#fps-counter")).to_be_visible()
        expect(page.locator("#channel-count")).to_be_visible()

    def test_sidebar_cards_present(self, page: Page) -> None:
        expect(page.locator("#coverage-title")).to_have_text("Surgical Target Coverage")
        expect(page.locator("#move-title")).to_have_text("Move Instruction")
        expect(page.locator("#high-gamma-title")).to_have_text("High Gamma")

    def test_canvases_exist(self, page: Page) -> None:
        expect(page.locator("#neural-canvas")).to_be_visible()
        expect(page.locator("#timeseries-canvas")).to_be_visible()


class TestWebSocket:
    """Verify WebSocket connection and data flow."""

    def test_connected_status(self, page: Page) -> None:
        expect(page.locator("#status-text")).to_have_text("Connected")

    def test_no_console_errors(self, page: Page) -> None:
        errors: list[str] = []
        page.on(
            "console",
            lambda msg: errors.append(msg.text) if msg.type == "error" else None,
        )
        # Wait a bit for any errors to surface
        page.wait_for_timeout(2_000)
        assert errors == [], f"Console errors: {errors}"

    def test_channel_count(self, page: Page) -> None:
        expect(page.locator("#channel-count")).to_have_text(
            re.compile(r"1024 channels")
        )


class TestDataStreaming:
    """Verify that data is actively streaming and updating the UI."""

    def test_time_advances(self, page: Page) -> None:
        """Time display should increase over a short interval."""
        time_el = page.locator("#time-display")

        t1_text = time_el.text_content()
        assert t1_text is not None
        t1 = float(t1_text.split("=")[1].strip().rstrip("s"))

        page.wait_for_timeout(2_000)

        t2_text = time_el.text_content()
        assert t2_text is not None
        t2 = float(t2_text.split("=")[1].strip().rstrip("s"))

        assert t2 > t1, f"Time did not advance: {t1} -> {t2}"

    def test_fps_nonzero(self, page: Page) -> None:
        page.wait_for_timeout(BASE_WAIT)
        fps_text = page.locator("#fps-counter").text_content()
        assert fps_text is not None
        fps = int(fps_text.split()[0])
        assert fps > 0, f"FPS should be > 0, got {fps}"

    def test_coverage_updates(self, page: Page) -> None:
        """Coverage should show a percentage value."""
        page.wait_for_timeout(BASE_WAIT)
        expect(page.locator("#coverage-display")).to_have_text(
            re.compile(r"Coverage: \d+\.\d+%")
        )

    def test_direction_shows_valid_label(self, page: Page) -> None:
        """Direction should be one of the known labels."""
        valid = {
            "Center",
            "Left",
            "Right",
            "Anterior",
            "Posterior",
            "Anterior-Left",
            "Anterior-Right",
            "Posterior-Left",
            "Posterior-Right",
        }
        page.wait_for_timeout(BASE_WAIT)
        direction = page.locator("#direction-display").text_content()
        assert direction in valid, f"Unexpected direction: {direction!r}"


class TestCanvasRendering:
    """Verify canvases have rendered content (non-blank)."""

    def test_heatmap_has_content(self, page: Page) -> None:
        page.wait_for_timeout(BASE_WAIT)
        data_len = page.evaluate(
            "document.getElementById('neural-canvas').toDataURL().length"
        )
        # A blank 544x544 canvas encodes to ~2000 chars; rendered heatmap >> 10000
        assert data_len > 10_000, (
            f"Heatmap canvas appears blank (dataURL len={data_len})"
        )

    def test_timeseries_has_content(self, page: Page) -> None:
        page.wait_for_timeout(BASE_WAIT)
        data_len = page.evaluate(
            "document.getElementById('timeseries-canvas').toDataURL().length"
        )
        assert data_len > 5_000, (
            f"Timeseries canvas appears blank (dataURL len={data_len})"
        )

    def test_heatmap_canvas_sized(self, page: Page) -> None:
        dims = page.evaluate(
            "JSON.stringify({w: document.getElementById('neural-canvas').width,"
            " h: document.getElementById('neural-canvas').height})"
        )
        import json

        d = json.loads(dims)
        assert d["w"] > 100, f"Heatmap width too small: {d['w']}"
        assert d["h"] > 100, f"Heatmap height too small: {d['h']}"


class TestHealthEndpoint:
    """Verify the /health API endpoint."""

    def test_health_ok(self, page: Page, base_url: str) -> None:
        resp = page.request.get(f"{base_url}/health")
        assert resp.ok
        data = resp.json()
        assert data["status"] == "ok"
        assert data["upstream_connected"] is True
