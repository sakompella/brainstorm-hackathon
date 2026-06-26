import type { FrameAnalysis, ValueRange } from "../types";
import { MAGMA, valueToColorIndex } from "../colormap";

export interface HeatmapDrawInput {
  ctx: CanvasRenderingContext2D;
  width: number; // CSS pixels
  height: number; // CSS pixels
  heatmap: number[][];
  centroid: [number, number]; // [row, col]
  range: ValueRange;
  analysis: FrameAnalysis;
  coverage: number;
}

/**
 * Draw the full heatmap frame: cells, grid overlay, border, centroid,
 * and guidance arrow.
 *
 * Ported from renderHeatmap() in frontend-old/app.js:221-408.
 * All direction/alignment logic has been moved to analysis.ts —
 * this function only draws based on pre-computed FrameAnalysis.
 */
export function drawHeatmap(input: HeatmapDrawInput): void {
  const { ctx, width, height, heatmap, centroid, range, analysis } = input;
  const { isAligned } = analysis;

  const rows = heatmap.length;
  const cols = heatmap[0].length;

  // Layout math
  const size = Math.min(width, height);
  const padding = Math.max(4, Math.floor(size * 0.04));
  const plotSize = Math.max(1, (size - 2 * padding) * 0.85);
  const cellSize = plotSize / cols;

  // Clear
  ctx.fillStyle = "#0a0a0f";
  ctx.fillRect(0, 0, width, height);

  // Centroid offset: center the hotspot at canvas center
  const [cy, cx] = centroid;
  const centerX = width / 2;
  const centerY = height / 2;
  const offsetX = centerX - (padding + (cx + 0.5) * cellSize);
  const offsetY = centerY - (padding + (cy + 0.5) * cellSize);

  // Draw cells
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const colorIdx = valueToColorIndex(heatmap[row][col], range);
      const [r, g, b] = MAGMA[colorIdx];
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(
        padding + col * cellSize + offsetX,
        padding + row * cellSize + offsetY,
        cellSize + 0.5,
        cellSize + 0.5,
      );
    }
  }

  // Grid overlay
  ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  for (let i = 0; i <= cols; i++) {
    const x = padding + i * cellSize + offsetX;
    ctx.moveTo(x, padding + offsetY);
    ctx.lineTo(x, padding + plotSize + offsetY);
  }
  for (let i = 0; i <= rows; i++) {
    const y = padding + i * cellSize + offsetY;
    ctx.moveTo(padding + offsetX, y);
    ctx.lineTo(padding + plotSize + offsetX, y);
  }
  ctx.stroke();

  // Border
  ctx.strokeStyle = isAligned
    ? "rgba(76, 222, 128, 0.8)"
    : "rgba(89, 224, 255, 0.5)";
  ctx.lineWidth = isAligned ? 12 : 4;
  ctx.strokeRect(padding + offsetX, padding + offsetY, plotSize, plotSize);

  // Centroid circle (always at canvas center)
  ctx.strokeStyle = isAligned ? "rgba(76, 222, 128, 0.9)" : "cyan";
  ctx.lineWidth = isAligned ? 12 : 4;
  ctx.beginPath();
  ctx.arc(centerX, centerY, Math.max(15, cellSize * 1.5), 0, 2 * Math.PI);
  ctx.stroke();

  // Guidance arrow (from array center to centroid).
  // Center matches the backend's definition: (n - 1) / 2.
  const arrayCenterRow = (rows - 1) / 2;
  const arrayCenterCol = (cols - 1) / 2;
  const vecX = cx - arrayCenterCol;
  const vecY = cy - arrayCenterRow;
  const mag = Math.sqrt(vecX ** 2 + vecY ** 2);

  if (mag > 1.5) {
    const arrayCenterX =
      padding + (arrayCenterCol + 0.5) * cellSize + offsetX;
    const arrayCenterY =
      padding + (arrayCenterRow + 0.5) * cellSize + offsetY;

    // Line from array center to canvas center
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(arrayCenterX, arrayCenterY);
    ctx.lineTo(centerX, centerY);
    ctx.stroke();

    // Arrowhead at midpoint
    const midX = (arrayCenterX + centerX) / 2;
    const midY = (arrayCenterY + centerY) / 2;
    const arrowSize = 20;
    const angle = Math.atan2(centerY - arrayCenterY, centerX - arrayCenterX);

    ctx.fillStyle = "lime";
    ctx.beginPath();
    ctx.moveTo(midX, midY);
    ctx.lineTo(
      midX - arrowSize * Math.cos(angle - Math.PI / 6),
      midY - arrowSize * Math.sin(angle - Math.PI / 6),
    );
    ctx.lineTo(
      midX - arrowSize * Math.cos(angle + Math.PI / 6),
      midY - arrowSize * Math.sin(angle + Math.PI / 6),
    );
    ctx.closePath();
    ctx.fill();
  }
}
