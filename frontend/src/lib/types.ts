// ── WebSocket messages (server → browser) ───────────────────────────

/** Sent once per connection with grid metadata. */
export interface InitMessage {
  type: "init";
  channels_coords: [number, number][];
  grid_size: number;
  fs: number;
  batch_size: number;
}

/** Lifecycle event when upstream connection state changes. */
export interface StatusMessage {
  type: "status";
  upstream_connected: boolean;
  upstream_state: UpstreamState;
  total_samples: number;
}

/** Processed neural features, emitted at ~20 Hz. */
export interface FeaturesMessage {
  type: "features";
  t: number;
  fs: number;
  n_ch: number;
  heatmap: number[][];
  centroid: [number, number]; // [row, col], 0-indexed
  center_distance: number; // 0 = corner, 1 = center
  confidence: number; // 1.0 if upstream connected
  bad_channels: number;
  total_samples: number;
}

/** Raw sample batch (only in --no-process / passthrough mode). */
export interface SampleBatchMessage {
  type: "sample_batch";
  neural_data: number[][];
  start_time_s: number;
  sample_count: number;
  fs: number;
}

export type ServerMessage =
  | InitMessage
  | StatusMessage
  | FeaturesMessage
  | SampleBatchMessage;

export type UpstreamState =
  | "connecting"
  | "connected"
  | "disconnected"
  | "ended";

// ── Derived domain types (internal, not from wire) ──────────────────

/** Connection state for the UI status bar. */
export type ConnectionStatus = "connected" | "connecting" | "disconnected";

/** Result of analyzing a features frame for display. */
export interface FrameAnalysis {
  direction: Direction;
  isAligned: boolean;
}

export type Direction =
  | "Center"
  | "Right"
  | "Left"
  | "Anterior"
  | "Posterior"
  | "Anterior-Right"
  | "Anterior-Left"
  | "Posterior-Right"
  | "Posterior-Left";

/** Auto-scaled colormap range, EMA-smoothed across frames. */
export interface ValueRange {
  vMin: number;
  vMax: number;
}
