import type { Direction, FrameAnalysis, ValueRange } from "./types";

const DB_EPS = 1e-12;

/**
 * Classify the direction from array center to centroid.
 * Ported from renderHeatmap() in frontend-old/app.js:335-380.
 *
 * @param centroid - [row, col] from backend (0-indexed)
 * @param gridRows - number of grid rows (typically 32)
 * @param gridCols - number of grid cols (typically 32)
 */
export function classifyDirection(
  centroid: [number, number],
  gridRows: number,
  gridCols: number,
): { direction: Direction; magnitude: number } {
  const [cy, cx] = centroid;
  // Geometric center matches the backend's definition: (n - 1) / 2.
  const centerRow = (gridRows - 1) / 2;
  const centerCol = (gridCols - 1) / 2;
  const vecX = cx - centerCol;
  const vecY = cy - centerRow;
  const mag = Math.sqrt(vecX ** 2 + vecY ** 2);

  const CENTER_THRESHOLD = 1.5;
  if (mag <= CENTER_THRESHOLD) {
    return { direction: "Center", magnitude: mag };
  }

  const unitX = vecX / mag;
  const unitY = vecY / mag;

  let direction: Direction;
  if (Math.abs(unitY) < 0.3 && unitX > 0.3) direction = "Right";
  else if (Math.abs(unitY) < 0.3 && unitX < -0.3) direction = "Left";
  else if (Math.abs(unitX) < 0.3 && unitY > 0.3) direction = "Posterior";
  else if (Math.abs(unitX) < 0.3 && unitY < -0.3) direction = "Anterior";
  else if (unitX > 0 && unitY < 0) direction = "Anterior-Right";
  else if (unitX < 0 && unitY < 0) direction = "Anterior-Left";
  else if (unitX > 0 && unitY > 0) direction = "Posterior-Right";
  else direction = "Posterior-Left";

  return { direction, magnitude: mag };
}

/**
 * Check if the array is well-aligned with the surgical target.
 * Ported from renderHeatmap() in frontend-old/app.js:303-314.
 *
 * @param centroidMagnitude - distance from array center to centroid in grid units
 * @param coverage - center_distance value from backend (0–1)
 */
export function checkAlignment(
  centroidMagnitude: number,
  coverage: number,
): boolean {
  return centroidMagnitude < 3 && coverage * 100 > 70;
}

/**
 * Analyze a features frame: produces direction + alignment for display.
 * Combines classifyDirection and checkAlignment.
 */
export function analyzeFrame(
  centroid: [number, number],
  gridRows: number,
  gridCols: number,
  coverage: number,
): FrameAnalysis {
  const { direction, magnitude } = classifyDirection(
    centroid,
    gridRows,
    gridCols,
  );
  const isAligned = checkAlignment(magnitude, coverage);
  return { direction, isAligned };
}

/**
 * Convert mean power to dB relative to a baseline.
 * Ported from ws.onmessage in frontend-old/app.js:680-686.
 */
export function toDb(meanPower: number, baselinePower: number): number {
  return (
    10 * Math.log10(Math.max(meanPower, DB_EPS) / Math.max(baselinePower, DB_EPS))
  );
}

/**
 * Compute mean power over a 2D heatmap.
 * Ported from ws.onmessage in frontend-old/app.js:672-679.
 */
export function meanHeatmapPower(heatmap: number[][]): number {
  let sum = 0;
  let count = 0;
  for (const row of heatmap) {
    for (const v of row) {
      sum += v;
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

/**
 * Update EMA-smoothed value range from a new heatmap frame.
 * Ported from renderHeatmap() in frontend-old/app.js:231-242.
 * Returns a new ValueRange (does not mutate input).
 *
 * @param prev - previous range
 * @param heatmap - current frame's 2D data
 * @param alpha - smoothing factor (0.1 in original)
 */
export function updateValueRange(
  prev: ValueRange,
  heatmap: number[][],
  alpha: number = 0.1,
): ValueRange {
  let maxVal = 0;
  let minVal = Infinity;
  for (const row of heatmap) {
    for (const v of row) {
      if (v > maxVal) maxVal = v;
      if (v < minVal) minVal = v;
    }
  }
  return {
    vMin: (1 - alpha) * prev.vMin + alpha * minVal,
    vMax: (1 - alpha) * prev.vMax + alpha * Math.max(maxVal, 0.001),
  };
}
