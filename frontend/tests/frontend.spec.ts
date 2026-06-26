/**
 * Neural Data Viewer — Frontend Regression Tests
 *
 * Mirrors all 17 assertions from scripts/test_frontend.sh plus additional
 * adversarial checks discovered by QA agents.
 *
 * Prerequisites: backend running on :8000, streamer on :8765.
 * Start with: ./start_all.sh  (or see AGENTS.md for manual startup)
 */

import { test, expect } from "@playwright/test";

// ---------------------------------------------------------------------------
// Helper: wait until the WebSocket is connected and at least one features
// frame has been delivered (t > 0).
// ---------------------------------------------------------------------------
async function waitForStreamData(page: import("@playwright/test").Page) {
  await page.waitForFunction(
    () => {
      const accent = document.querySelectorAll(".accent")[0];
      if (!accent) return false;
      const text = accent.textContent ?? "";
      const match = text.match(/t = ([\d.]+)s/);
      return match !== null && parseFloat(match[1]) > 0;
    },
    undefined,
    { timeout: 10_000 }
  );
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await waitForStreamData(page);
});

// ── 1. Page load ─────────────────────────────────────────────────────────────

test.describe("Page load", () => {
  test("page title is 'Neural Data Viewer'", async ({ page }) => {
    await expect(page).toHaveTitle("Neural Data Viewer");
  });

  test("main heading is 'Neural Activity'", async ({ page }) => {
    // The first h1 in the document is the app header
    const firstH1 = page.locator("h1").first();
    await expect(firstH1).toHaveText("Neural Activity");
  });

  test("exactly 2 canvas elements", async ({ page }) => {
    const canvases = page.locator("canvas");
    await expect(canvases).toHaveCount(2);
  });
});

// ── 2. WebSocket connection ───────────────────────────────────────────────────

test.describe("WebSocket connection", () => {
  test("status indicator has class 'connected'", async ({ page }) => {
    const indicator = page.locator(".status-indicator");
    await expect(indicator).toHaveClass(/connected/);
  });

  test("status text reads 'Connected'", async ({ page }) => {
    // Second <span> inside .status-bar
    const statusText = page.locator(".status-bar span").nth(1);
    await expect(statusText).toHaveText("Connected");
  });
});

// ── 3. Data streaming ────────────────────────────────────────────────────────

test.describe("Data streaming", () => {
  test("time display matches format 't = X.XXs'", async ({ page }) => {
    const timeEl = page.locator(".accent").first();
    await expect(timeEl).toHaveText(/^t = \d+\.\d+s$/);
  });

  test("time advances over 2 seconds", async ({ page }) => {
    const timeEl = page.locator(".accent").first();

    const t1Text = await timeEl.textContent();
    const t1 = parseFloat((t1Text ?? "").replace("t = ", "").replace("s", ""));

    // Wait 2 s then sample again
    await page.waitForTimeout(2000);

    const t2Text = await timeEl.textContent();
    const t2 = parseFloat((t2Text ?? "").replace("t = ", "").replace("s", ""));

    expect(t2).toBeGreaterThan(t1);
  });
});

// ── 4. Heatmap canvas ────────────────────────────────────────────────────────

test.describe("Heatmap canvas", () => {
  test("heatmap canvas has non-zero width", async ({ page }) => {
    const width: number = await page.evaluate(
      () => (document.querySelectorAll("canvas")[0] as HTMLCanvasElement).width
    );
    expect(width).toBeGreaterThan(0);
  });

  test("heatmap canvas has non-zero height", async ({ page }) => {
    const height: number = await page.evaluate(
      () => (document.querySelectorAll("canvas")[0] as HTMLCanvasElement).height
    );
    expect(height).toBeGreaterThan(0);
  });

  test("heatmap canvas is square (width === height)", async ({ page }) => {
    const [width, height]: [number, number] = await page.evaluate(() => {
      const c = document.querySelectorAll("canvas")[0] as HTMLCanvasElement;
      return [c.width, c.height];
    });
    expect(width).toBe(height);
  });
});

// ── 5. Time series canvas ────────────────────────────────────────────────────

test.describe("Time series canvas", () => {
  test("time series canvas has non-zero height", async ({ page }) => {
    const height: number = await page.evaluate(
      () => (document.querySelectorAll("canvas")[1] as HTMLCanvasElement).height
    );
    expect(height).toBeGreaterThan(0);
  });

  test("time series canvas height < 1000 (regression: was 17000px)", async ({
    page,
  }) => {
    const height: number = await page.evaluate(
      () => (document.querySelectorAll("canvas")[1] as HTMLCanvasElement).height
    );
    expect(height).toBeLessThan(1000);
  });
});

// ── 6. Centering card ────────────────────────────────────────────────────────

test.describe("Centering card", () => {
  test("centering text matches format 'X.X% centered'", async ({ page }) => {
    const body = page.locator(".card-coverage .card-body");
    await expect(body).toHaveText(/^\d+\.\d+% centered$/);
  });

  test("centering card body is non-empty", async ({ page }) => {
    const body = page.locator(".card-coverage .card-body");
    const text = await body.textContent();
    expect(text).toBeTruthy();
  });
});

// ── 7. Direction card ────────────────────────────────────────────────────────

test.describe("Direction card", () => {
  test("direction is a valid direction enum value", async ({ page }) => {
    const validDirections = [
      "Center",
      "Right",
      "Left",
      "Anterior",
      "Posterior",
      "Anterior-Right",
      "Anterior-Left",
      "Posterior-Right",
      "Posterior-Left",
    ];
    const direction = await page.locator("#direction-display").textContent();
    expect(validDirections).toContain(direction?.trim());
  });
});

// ── 8. FPS and channel count ─────────────────────────────────────────────────

test.describe("FPS and channel count", () => {
  test("FPS display matches format 'N FPS'", async ({ page }) => {
    const fpsEl = page.locator(".accent").nth(1);
    await expect(fpsEl).toHaveText(/^\d+ FPS$/);
  });

  test("channel count contains '1024 channels'", async ({ page }) => {
    const chEl = page.locator(".accent").nth(2);
    await expect(chEl).toContainText("1024 channels");
  });
});

// ── Adversarial tests (QA-discovered) ────────────────────────────────────────

test.describe("Adversarial / QA checks", () => {
  // Known issue: CenteringCard and MoveCard both render <h1> wrappers, so the
  // page has 3 h1 elements instead of 1. Tracked here; fix before removing fixme.
  test.fixme(
    "only one h1 element on the page",
    async ({ page }) => {
      const count: number = await page.evaluate(
        () => document.querySelectorAll("h1").length
      );
      expect(count).toBe(1);
    }
  );

  test("heatmap center pixel is non-black (canvas is actually drawing)", async ({
    page,
  }) => {
    // Wait a moment extra for paint to settle
    await page.waitForTimeout(500);

    const isNonBlack: boolean = await page.evaluate(() => {
      const canvas = document.querySelectorAll("canvas")[0] as HTMLCanvasElement;
      const ctx = canvas.getContext("2d");
      if (!ctx) return false;
      const cx = Math.floor(canvas.width / 2);
      const cy = Math.floor(canvas.height / 2);
      const pixel = ctx.getImageData(cx, cy, 1, 1).data;
      // pixel[3] is alpha; if alpha=0 or all RGB=0 treat as black/empty
      return pixel[3] > 0 && (pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0);
    });

    expect(isNonBlack).toBe(true);
  });

  test("no horizontal overflow (scrollWidth <= clientWidth)", async ({
    page,
  }) => {
    const hasOverflow: boolean = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth
    );
    expect(hasOverflow).toBe(false);
  });
});
