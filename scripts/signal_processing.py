"""
Signal processing for neural data streams.

Ported from signal_processing.ipynb - full pipeline with:
- Bad channel detection (dead/artifact/saturated)
- 60 Hz notch filter (line noise)
- 70-150 Hz bandpass (high gamma)
- Power extraction with EMA smoothing
- Spatial smoothing (gaussian)
- Drift tracking algorithms (elastic lasso, convex hull, bounding box, sliding template)
"""

from collections import deque
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos
from scipy.spatial import ConvexHull


class ProcessingSettings(TypedDict):
    notch_60hz: bool
    bandpass: bool
    bad_channel_detection: bool
    spatial_sigma: float
    ema_alpha: float


DIFFICULTY_SETTINGS: dict[str, ProcessingSettings] = {
    "super_easy": {
        "notch_60hz": False,
        "bandpass": True,
        "bad_channel_detection": False,
        "spatial_sigma": 0.0,
        "ema_alpha": 0.3,
    },
    "easy": {
        "notch_60hz": False,
        "bandpass": True,
        "bad_channel_detection": False,
        "spatial_sigma": 0.5,
        "ema_alpha": 0.2,
    },
    "medium": {
        "notch_60hz": True,
        "bandpass": True,
        "bad_channel_detection": True,
        "spatial_sigma": 1.0,
        "ema_alpha": 0.15,
    },
    "hard": {
        "notch_60hz": True,
        "bandpass": True,
        "bad_channel_detection": True,
        "spatial_sigma": 1.5,
        "ema_alpha": 0.1,
    },
}


@dataclass(slots=True)
class BadChannelInfo:
    dead: np.ndarray
    artifact: np.ndarray
    saturated: np.ndarray
    variance: np.ndarray


class BadChannelDetector:
    """Detect dead, artifact, and saturated channels."""

    def __init__(self, dead_thresh: float = 1e-8, artifact_mult: float = 50):
        self.dead_thresh = dead_thresh
        self.artifact_mult = artifact_mult

    def detect(self, data_buffer: np.ndarray) -> tuple[np.ndarray, BadChannelInfo]:
        """Detect bad channels from data buffer.

        Args:
            data_buffer: (n_samples, n_channels) array

        Returns:
            bad_mask: boolean array (n_channels,)
            info: BadChannelInfo with details
        """
        variance = np.var(data_buffer, axis=0)
        median_var = np.median(variance)

        dead = variance < self.dead_thresh
        artifact = variance > median_var * self.artifact_mult
        ranges = np.ptp(data_buffer, axis=0)
        saturated = ranges < self.dead_thresh

        bad_mask = dead | artifact | saturated

        return bad_mask, BadChannelInfo(
            dead=np.where(dead)[0],
            artifact=np.where(artifact)[0],
            saturated=np.where(saturated)[0],
            variance=variance,
        )


class SignalFilter:
    """Streaming-compatible signal filter with state.

    Applies optional 60Hz notch and 70-150Hz bandpass.
    """

    def __init__(
        self,
        fs: float = 500,
        notch_freq: float = 60,
        notch_q: float = 30,
        lowcut: float = 70,
        highcut: float = 150,
        order: int = 4,
        use_notch: bool = True,
        use_bandpass: bool = True,
    ):
        self.fs = fs
        self.use_notch = use_notch
        self.use_bandpass = use_bandpass

        if use_notch:
            b, a = iirnotch(notch_freq, notch_q, fs)
            self.notch_sos = tf2sos(b, a)
        else:
            self.notch_sos = None

        if use_bandpass:
            self.bp_sos = butter(
                order, [lowcut, highcut], btype="band", fs=fs, output="sos"
            )
        else:
            self.bp_sos = None

        self.notch_zi: np.ndarray | None = None
        self.bp_zi: np.ndarray | None = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Process a chunk of data, maintaining filter state.

        Args:
            chunk: (n_samples, n_channels) array

        Returns:
            filtered: (n_samples, n_channels) array
        """
        n_channels = chunk.shape[1]

        # Initialize states if needed
        if self.use_notch and self.notch_zi is None:
            zi = sosfilt_zi(self.notch_sos)
            self.notch_zi = np.tile(zi[:, :, np.newaxis], (1, 1, n_channels))

        if self.use_bandpass and self.bp_zi is None:
            zi = sosfilt_zi(self.bp_sos)
            self.bp_zi = np.tile(zi[:, :, np.newaxis], (1, 1, n_channels))

        # Apply notch filter
        if self.use_notch:
            notched, self.notch_zi = sosfilt(
                self.notch_sos, chunk, zi=self.notch_zi, axis=0
            )
        else:
            notched = chunk

        # Apply bandpass filter
        if self.use_bandpass:
            filtered, self.bp_zi = sosfilt(self.bp_sos, notched, zi=self.bp_zi, axis=0)
        else:
            filtered = notched

        return filtered


class PowerExtractor:
    """Extract power envelope with dual EMA for baseline normalization."""

    def __init__(self, alpha_fast: float = 0.15, alpha_base: float = 0.005):
        self.alpha_fast = alpha_fast
        self.alpha_base = alpha_base
        self.p_fast: np.ndarray | None = None
        self.p_base: np.ndarray | None = None

    def process(self, filtered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract smoothed power from filtered data.

        Args:
            filtered: (n_samples, n_channels) array

        Returns:
            power: (n_channels,) raw smoothed power
            normalized: (n_channels,) log-normalized (fast vs baseline)
        """
        power_instant = filtered**2
        power = np.mean(power_instant, axis=0)

        if self.p_fast is None:
            self.p_fast = power.copy()
            self.p_base = power.copy()
        else:
            self.p_fast = self.alpha_fast * power + (1 - self.alpha_fast) * self.p_fast
            self.p_base = self.alpha_base * power + (1 - self.alpha_base) * self.p_base

        # Log-normalized: positive = above baseline, negative = below
        normalized = np.log(self.p_fast + 1e-12) - np.log(self.p_base + 1e-12)

        return self.p_fast.copy(), normalized


def to_grid(
    power_1024: np.ndarray,
    bad_mask: np.ndarray | None = None,
    sigma: float = 0.0,
    grid_size: int = 32,
) -> np.ndarray:
    """Convert 1024 channel values to 32x32 grid with optional smoothing.

    Args:
        power_1024: (n_channels,) power array
        bad_mask: (n_channels,) boolean mask of bad channels
        sigma: gaussian smoothing sigma (0 = no smoothing)
        grid_size: grid dimension (default 32)

    Returns:
        grid: (grid_size, grid_size) array
    """
    grid = power_1024.reshape(grid_size, grid_size)

    if bad_mask is not None:
        bad_grid = bad_mask.reshape(grid_size, grid_size)
        grid = grid.copy()
        grid[bad_grid] = 0

    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)

    return grid


_COORD_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _get_coord_grids(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Get cached coordinate grids to avoid allocation per call."""
    key = (h, w)
    if key not in _COORD_CACHE:
        _COORD_CACHE[key] = np.mgrid[0:h, 0:w]
    return _COORD_CACHE[key]


def weighted_centroid(grid: np.ndarray, threshold_percentile: float = 50) -> np.ndarray:
    """Find center of mass weighted by intensity.

    Args:
        grid: (H, W) activity grid
        threshold_percentile: zero out values below this percentile

    Returns:
        center: [row, col] array
    """
    threshold = np.percentile(grid, threshold_percentile)
    masked = np.where(grid > threshold, grid, 0)

    total = masked.sum()
    if total < 1e-10:
        return np.array([grid.shape[0] / 2, grid.shape[1] / 2], dtype=np.float32)

    rows, cols = _get_coord_grids(grid.shape[0], grid.shape[1])
    center_row = (rows * masked).sum() / total
    center_col = (cols * masked).sum() / total
    return np.array([center_row, center_col], dtype=np.float32)


def compute_presence(power: np.ndarray, k: int = 50) -> float:
    """Compute global presence indicator from power.

    Args:
        power: per-channel power array
        k: number of top channels to average

    Returns:
        mean power of top-k channels
    """
    topk = np.partition(power.ravel(), -k)[-k:]
    return float(np.mean(topk))


# =============================================================================
# DRIFT TRACKING ALGORITHMS
# Ported from signal_processing.ipynb with exact settings
# =============================================================================


class ElasticLassoTracker:
    """Adaptive lasso that wraps hotspots - slow to expand, fast to shrink.

    Settings from notebook: expand=0.15, contract=0.5, decay_frames=5
    """

    def __init__(
        self,
        grid_size: int = 32,
        activity_threshold_pct: float = 75,
        decay_frames: int = 5,
        contract_rate: float = 0.5,
        expand_rate: float = 0.15,
        display_thresh: float = 0.5,
    ):
        self.grid_size = grid_size
        self.activity_threshold_pct = activity_threshold_pct
        self.decay_frames = decay_frames
        self.contract_rate = contract_rate
        self.expand_rate = expand_rate
        self.display_thresh = display_thresh

        self.membership = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.inactive_frames = np.zeros((grid_size, grid_size), dtype=np.float32)

    def update(self, grid: np.ndarray) -> dict:
        """Update lasso with new grid and return centroid + boundary."""
        threshold = np.percentile(grid, self.activity_threshold_pct)
        active = (grid > threshold).astype(np.float32)

        # Update inactive counter
        self.inactive_frames = np.where(active > 0.5, 0, self.inactive_frames + 1)

        # EXPAND slowly where active
        self.membership = np.where(
            active > 0.5,
            self.membership + self.expand_rate * (1 - self.membership),
            self.membership,
        )

        # CONTRACT fast where inactive too long
        decay_mask = self.inactive_frames > self.decay_frames
        self.membership = np.where(
            decay_mask,
            self.membership * (1 - self.contract_rate),
            self.membership,
        )

        self.membership = np.clip(self.membership, 0, 1)

        # Compute centroid
        total = self.membership.sum()
        if total > 1e-10:
            rows, cols = _get_coord_grids(self.grid_size, self.grid_size)
            cr = float((rows * self.membership).sum() / total)
            cc = float((cols * self.membership).sum() / total)
        else:
            cr, cc = self.grid_size / 2, self.grid_size / 2

        # Boundary mask for visualization
        boundary_mask = (self.membership > self.display_thresh).astype(np.float32)

        return {
            "centroid": [cr, cc],
            "membership": self.membership.copy(),
            "boundary_mask": boundary_mask,
            "area": float(boundary_mask.sum()),
        }


class ConvexHullTracker:
    """Track convex hull of persistent hotspots.

    Settings from notebook: persistence_frames=3, activity_threshold_pct=70
    """

    def __init__(
        self,
        grid_size: int = 32,
        activity_threshold_pct: float = 70,
        persistence_frames: int = 3,
    ):
        self.grid_size = grid_size
        self.activity_threshold_pct = activity_threshold_pct
        self.persistence_frames = persistence_frames

        self.peak_persistence = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.last_centroid = np.array([grid_size / 2, grid_size / 2], dtype=np.float32)

    def update(self, grid: np.ndarray) -> dict:
        """Update hull tracker with new grid."""
        local_max = maximum_filter(grid, size=3) == grid
        threshold = np.percentile(grid, self.activity_threshold_pct)
        peaks = local_max & (grid > threshold)

        # Faster persistence buildup (+2), slower decay (-0.3)
        self.peak_persistence = np.where(
            peaks,
            np.minimum(self.peak_persistence + 2, self.persistence_frames * 3),
            np.maximum(self.peak_persistence - 0.3, 0),
        )

        persistent_mask = self.peak_persistence >= self.persistence_frames
        persistent_coords = np.argwhere(persistent_mask)

        hull_vertices: list[list[float]] = []

        if len(persistent_coords) >= 3:
            try:
                hull = ConvexHull(persistent_coords)
                hull_pts = persistent_coords[hull.vertices]
                cr = float(hull_pts[:, 0].mean())
                cc = float(hull_pts[:, 1].mean())
                self.last_centroid = np.array([cr, cc], dtype=np.float32)
                hull_vertices = hull_pts.tolist()
            except Exception:
                if len(persistent_coords) > 0:
                    cr = float(persistent_coords[:, 0].mean())
                    cc = float(persistent_coords[:, 1].mean())
                    self.last_centroid = np.array([cr, cc], dtype=np.float32)
        elif len(persistent_coords) >= 1:
            cr = float(persistent_coords[:, 0].mean())
            cc = float(persistent_coords[:, 1].mean())
            self.last_centroid = np.array([cr, cc], dtype=np.float32)

        return {
            "centroid": self.last_centroid.tolist(),
            "hull_vertices": hull_vertices,
            "n_persistent_peaks": int(persistent_mask.sum()),
        }


class BoundingBoxTracker:
    """Adaptive bounding box that shrinks toward activity, expands to include new.

    Settings from notebook: shrink_rate=0.15, expand_rate=0.2
    """

    def __init__(
        self,
        grid_size: int = 32,
        activity_threshold_pct: float = 80,
        shrink_rate: float = 0.15,
        expand_rate: float = 0.2,
    ):
        self.grid_size = grid_size
        self.activity_threshold_pct = activity_threshold_pct
        self.shrink_rate = shrink_rate
        self.expand_rate = expand_rate

        # Box bounds: [row_min, row_max, col_min, col_max]
        quarter = grid_size / 4
        self.box = [quarter, grid_size - quarter, quarter, grid_size - quarter]

    def update(self, grid: np.ndarray) -> dict:
        """Update bounding box with new grid."""
        threshold = np.percentile(grid, self.activity_threshold_pct)
        active = grid > threshold
        active_coords = np.argwhere(active)

        if len(active_coords) > 0:
            act_rmin = float(active_coords[:, 0].min())
            act_rmax = float(active_coords[:, 0].max())
            act_cmin = float(active_coords[:, 1].min())
            act_cmax = float(active_coords[:, 1].max())

            # EXPAND to include activity outside box
            if act_rmin < self.box[0]:
                self.box[0] -= self.expand_rate * (self.box[0] - act_rmin)
            if act_rmax > self.box[1]:
                self.box[1] += self.expand_rate * (act_rmax - self.box[1])
            if act_cmin < self.box[2]:
                self.box[2] -= self.expand_rate * (self.box[2] - act_cmin)
            if act_cmax > self.box[3]:
                self.box[3] += self.expand_rate * (act_cmax - self.box[3])

            # SHRINK toward activity extent
            self.box[0] += self.shrink_rate * (act_rmin - self.box[0])
            self.box[1] += self.shrink_rate * (act_rmax - self.box[1])
            self.box[2] += self.shrink_rate * (act_cmin - self.box[2])
            self.box[3] += self.shrink_rate * (act_cmax - self.box[3])

        # Clamp to valid range
        self.box = [
            max(0, self.box[0]),
            min(self.grid_size - 1, self.box[1]),
            max(0, self.box[2]),
            min(self.grid_size - 1, self.box[3]),
        ]

        cr = (self.box[0] + self.box[1]) / 2
        cc = (self.box[2] + self.box[3]) / 2

        return {
            "centroid": [cr, cc],
            "box": self.box.copy(),  # [row_min, row_max, col_min, col_max]
        }


class SlidingTemplateTracker:
    """Sliding window template - rolling average of last N frames.

    Settings from notebook: window_frames=20
    """

    def __init__(
        self,
        grid_size: int = 32,
        window_frames: int = 20,
    ):
        self.grid_size = grid_size
        self.window_frames = window_frames
        self.window: deque[np.ndarray] = deque(maxlen=window_frames)
        self.last_centroid = np.array([grid_size / 2, grid_size / 2], dtype=np.float32)

    def update(self, grid: np.ndarray) -> dict:
        """Update sliding template with new grid."""
        # Normalize grid
        max_val = np.percentile(grid, 99) + 1e-10
        grid_norm = grid / max_val
        self.window.append(grid_norm.copy())

        if len(self.window) > 0:
            template = np.mean(self.window, axis=0)
            threshold = np.percentile(template, 50)
            masked = np.where(template > threshold, template, 0)
            total = masked.sum()

            if total > 1e-10:
                rows, cols = _get_coord_grids(self.grid_size, self.grid_size)
                cr = float((rows * masked).sum() / total)
                cc = float((cols * masked).sum() / total)
                self.last_centroid = np.array([cr, cc], dtype=np.float32)

        return {
            "centroid": self.last_centroid.tolist(),
            "window_size": len(self.window),
        }


class DriftTracker:
    """Combined drift tracker running all algorithms in parallel."""

    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size
        self.elastic_lasso = ElasticLassoTracker(grid_size=grid_size)
        self.convex_hull = ConvexHullTracker(grid_size=grid_size)
        self.bounding_box = BoundingBoxTracker(grid_size=grid_size)
        self.sliding_template = SlidingTemplateTracker(grid_size=grid_size)

    def update(self, grid: np.ndarray) -> dict:
        """Update all trackers and return combined results."""
        lasso_result = self.elastic_lasso.update(grid)
        hull_result = self.convex_hull.update(grid)
        box_result = self.bounding_box.update(grid)
        template_result = self.sliding_template.update(grid)

        return {
            "elastic_lasso": lasso_result,
            "convex_hull": hull_result,
            "bounding_box": box_result,
            "sliding_template": template_result,
        }


# =============================================================================
# NEURAL PROCESSOR
# =============================================================================


class NeuralProcessor:
    """Complete neural processing pipeline with configurable settings."""

    def __init__(
        self,
        fs: float = 500,
        settings: ProcessingSettings | None = None,
        grid_size: int = 32,
        stateless: bool = False,
        enable_drift_tracking: bool = True,
    ):
        self.fs = fs
        self.grid_size = grid_size
        self.settings = settings or DIFFICULTY_SETTINGS["hard"]
        self.stateless = stateless
        self.enable_drift_tracking = enable_drift_tracking

        self.bad_detector = BadChannelDetector()
        self.signal_filter = SignalFilter(
            fs=fs,
            use_notch=self.settings["notch_60hz"],
            use_bandpass=self.settings["bandpass"],
        )
        # Fast alpha from settings, slow baseline alpha ~20x slower
        self.power_extractor = PowerExtractor(
            alpha_fast=self.settings["ema_alpha"],
            alpha_base=self.settings["ema_alpha"] / 20,
        )

        self.bad_mask: np.ndarray | None = None
        self.init_buffer: list[np.ndarray] = []
        self.init_samples_needed = int(1.0 * fs)  # 1 second
        self.do_bad_detection = self.settings["bad_channel_detection"]

        # Accumulator for stable centroid tracking (disabled in stateless mode)
        self.accumulated: np.ndarray | None = None
        self.accumulator_decay = 0.7
        self.accumulator_weight = 0.1

        # Light temporal smoothing for stateless mode (fast EMA, no baseline)
        self.smoothed_power: np.ndarray | None = None
        self.smooth_alpha = 0.15  # Higher = more responsive, lower = smoother

        # Drift tracking
        if enable_drift_tracking:
            self.drift_tracker = DriftTracker(grid_size=grid_size)
        else:
            self.drift_tracker = None

    def process_batch(self, neural_data: np.ndarray) -> dict | None:
        """Process a batch of neural data.

        Args:
            neural_data: (batch_size, n_channels) array

        Returns:
            dict with heatmap, centroid, presence, drift_tracking, etc. or None if still initializing
        """
        chunk = np.asarray(neural_data, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)

        n_channels = chunk.shape[1]

        # Bad channel detection phase
        if self.bad_mask is None:
            if self.do_bad_detection:
                self.init_buffer.append(chunk)
                total_samples = sum(c.shape[0] for c in self.init_buffer)

                if total_samples >= self.init_samples_needed:
                    all_data = np.vstack(self.init_buffer)
                    self.bad_mask, _info = self.bad_detector.detect(all_data)
                    self.init_buffer = []
                else:
                    return None
            else:
                self.bad_mask = np.zeros(n_channels, dtype=bool)

        # Filter
        filtered = self.signal_filter.process(chunk)

        if self.stateless:
            # Stateless mode: instantaneous power with light temporal smoothing
            power = np.mean(filtered**2, axis=0)

            # Apply light EMA to reduce frame-to-frame jitter
            if self.smoothed_power is None:
                self.smoothed_power = power.copy()
            else:
                self.smoothed_power = (
                    self.smooth_alpha * power
                    + (1 - self.smooth_alpha) * self.smoothed_power
                )

            grid = to_grid(
                self.smoothed_power,
                self.bad_mask,
                sigma=self.settings["spatial_sigma"],
                grid_size=self.grid_size,
            )
            centroid = weighted_centroid(grid, threshold_percentile=50)
            presence = compute_presence(self.smoothed_power)
        else:
            # Stateful mode: EMA + accumulation for stable tracking
            power, normalized = self.power_extractor.process(filtered)

            grid = to_grid(
                normalized,
                self.bad_mask,
                sigma=self.settings["spatial_sigma"],
                grid_size=self.grid_size,
            )

            # Update accumulator for stable tracking (use positive part for centroid)
            if self.accumulated is None:
                self.accumulated = np.zeros_like(grid)

            grid_positive = np.maximum(grid, 0)
            max_val = np.percentile(grid_positive, 99) + 1e-10
            grid_norm = np.clip(grid_positive / max_val, 0, 1)
            self.accumulated = (
                self.accumulator_decay * self.accumulated
                + self.accumulator_weight * grid_norm
            )

            centroid = weighted_centroid(self.accumulated, threshold_percentile=50)
            presence = compute_presence(power)

        # Run drift tracking algorithms
        drift_tracking = None
        if self.drift_tracker is not None:
            drift_tracking = self.drift_tracker.update(grid)

        return {
            "heatmap": grid,
            "centroid": centroid,
            "presence": presence,
            "bad_channels": int(self.bad_mask.sum())
            if self.bad_mask is not None
            else 0,
            "drift_tracking": drift_tracking,
        }


# Legacy API for backward compatibility with backend.py
@dataclass(frozen=True, slots=True)
class EMAState:
    """Immutable state for exponential moving average computation."""

    a_fast: float
    a_base: float
    p_fast: np.ndarray
    p_base: np.ndarray


def create_ema_state(
    n_ch: int,
    fs: float,
    tau_fast_s: float = 0.2,
    tau_base_s: float = 8.0,
) -> EMAState:
    """Factory function to create initial EMA state."""
    dt = 1.0 / fs
    return EMAState(
        a_fast=float(dt / tau_fast_s),
        a_base=float(dt / tau_base_s),
        p_fast=np.zeros((n_ch,), dtype=np.float32),
        p_base=np.zeros((n_ch,), dtype=np.float32),
    )


def update_ema(state: EMAState, batch: np.ndarray) -> EMAState:
    """Process batch of samples and return updated state (vectorized)."""
    n = len(batch)
    if n == 0:
        return state

    x2 = batch * batch  # (n, 1024) - vectorized square

    # Compute decay factors for the recurrence: newest sample has highest weight
    # p[n] = (1-a)^n * p[0] + a * sum_{i=0}^{n-1} (1-a)^{n-1-i} * x2[i]
    decay_fast = (1.0 - state.a_fast) ** np.arange(n - 1, -1, -1, dtype=np.float32)
    decay_base = (1.0 - state.a_base) ** np.arange(n - 1, -1, -1, dtype=np.float32)

    # Weighted contribution from new samples
    contrib_fast = state.a_fast * np.einsum("i,ij->j", decay_fast, x2)
    contrib_base = state.a_base * np.einsum("i,ij->j", decay_base, x2)

    # Decay existing state and add new contribution
    p_fast = ((1.0 - state.a_fast) ** n) * state.p_fast + contrib_fast
    p_base = ((1.0 - state.a_base) ** n) * state.p_base + contrib_base

    return EMAState(
        a_fast=state.a_fast,
        a_base=state.a_base,
        p_fast=p_fast.astype(np.float32),
        p_base=p_base.astype(np.float32),
    )


def ema_activity(state: EMAState) -> np.ndarray:
    """Compute activity level from state."""
    return np.sqrt(state.p_fast + 1e-12)
