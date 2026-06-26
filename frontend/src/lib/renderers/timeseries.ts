import type { TimeSeriesPoint } from "../timeseries-buffer";

export interface TimeSeriesDrawInput {
  ctx: CanvasRenderingContext2D;
  width: number; // CSS pixels
  height: number; // CSS pixels
  points: readonly TimeSeriesPoint[];
}

const PADDING = { top: 20, right: 20, bottom: 30, left: 50 } as const;

/**
 * Draw the EKG-style high-gamma time series plot.
 *
 * Ported from plotTimeSeries() in frontend-old/app.js:495-598.
 * Newest data is positioned at 70% of the plot width.
 */
export function drawTimeSeries(input: TimeSeriesDrawInput): void {
  const { ctx, width, height, points } = input;

  // Clear
  ctx.fillStyle = "#0a0a0f";
  ctx.fillRect(0, 0, width, height);

  if (points.length < 2) return;

  const plotWidth = width - PADDING.left - PADDING.right;
  const plotHeight = height - PADDING.top - PADDING.bottom;

  // Value range
  let minValue = Infinity;
  let maxValue = -Infinity;
  for (const p of points) {
    if (p.value < minValue) minValue = p.value;
    if (p.value > maxValue) maxValue = p.value;
  }
  const valueRange = maxValue - minValue || 1;

  // Time range: newest at 70% of plot width
  const latestTime = points[points.length - 1].t;
  const oldestTime = points[0].t;
  const actualTimeRange = latestTime - oldestTime || 1;
  const displayTimeRange = actualTimeRange / 0.7;

  // Axes
  ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PADDING.left, PADDING.top);
  ctx.lineTo(PADDING.left, height - PADDING.bottom);
  ctx.lineTo(width - PADDING.right, height - PADDING.bottom);
  ctx.stroke();

  // Horizontal grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = PADDING.top + (plotHeight * i) / 4;
    ctx.beginPath();
    ctx.moveTo(PADDING.left, y);
    ctx.lineTo(width - PADDING.right, y);
    ctx.stroke();
  }

  // Y-axis labels
  ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const value = maxValue - (valueRange * i) / 4;
    const y = PADDING.top + (plotHeight * i) / 4;
    ctx.fillText(value.toFixed(2), PADDING.left - 5, y);
  }

  // Data trace
  ctx.strokeStyle = "#ff6b6b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const timeFromLatest = latestTime - p.t;
    const normalizedPos = 0.7 - timeFromLatest / displayTimeRange;
    const x = PADDING.left + normalizedPos * plotWidth;
    const y =
      height - PADDING.bottom - ((p.value - minValue) / valueRange) * plotHeight;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  ctx.font = "12px JetBrains Mono, monospace";
  ctx.textAlign = "center";
  ctx.fillText("Time (s)", width / 2, height - 5);

  ctx.save();
  ctx.translate(10, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Mean Power (dB)", 0, 0);
  ctx.restore();
}
