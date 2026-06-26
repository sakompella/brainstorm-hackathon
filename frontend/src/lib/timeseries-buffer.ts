export interface TimeSeriesPoint {
  t: number;
  value: number;
}

/**
 * Bounded FIFO buffer for time series data.
 * Replaces the module-level timeSeriesData + maxTimeSeriesPoints from app.js.
 */
export class TimeSeriesBuffer {
  private data: TimeSeriesPoint[] = [];
  readonly capacity: number;

  constructor(capacity: number = 500) {
    this.capacity = capacity;
  }

  push(point: TimeSeriesPoint): void {
    this.data.push(point);
    if (this.data.length > this.capacity) {
      this.data.shift();
    }
  }

  get points(): readonly TimeSeriesPoint[] {
    return this.data;
  }

  get length(): number {
    return this.data.length;
  }

  clear(): void {
    this.data = [];
  }
}
