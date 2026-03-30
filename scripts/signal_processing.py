"""
Signal processing for neural data streams.

Ported from signal_processing.ipynb - full pipeline with:
- Bad channel detection (dead/artifact/saturated)
- 60 Hz notch filter (line noise)
- 70-150 Hz bandpass (high gamma)
- Power extraction with EMA smoothing
- Spatial smoothing (gaussian)
- Weighted centroid tracking
"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos


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

        if self.p_fast is None or self.p_base is None:
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
        rows, cols = np.mgrid[0:h, 0:w]
        _COORD_CACHE[key] = (rows, cols)
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


def compute_center_distance(centroid: np.ndarray, grid_size: int = 32) -> float:
    """
    Distance from centroid to grid center.

    Measures how well the hotspot centroid is centered on the grid, useful for
    guidance during array positioning. Range [0, 1] where 1 = perfectly centered,
    0 = at corner of grid.

    Args:
        centroid: [row, col] array with centroid position
        grid_size: grid dimension (default 32 for 32x32 array)

    Returns:
        centering: normalized metric 1.0 (perfectly centered) to 0.0 (corner)
    """
    center = grid_size / 2
    raw = np.sqrt((centroid[0] - center) ** 2 + (centroid[1] - center) ** 2)
    max_dist = np.sqrt(2) * center
    return float(1 - (raw / max_dist))


class NeuralProcessor:
    """Complete neural processing pipeline with configurable settings."""

    def __init__(
        self,
        fs: float = 500,
        settings: ProcessingSettings | None = None,
        grid_size: int = 32,
        stateless: bool = False,
    ):
        self.fs = fs
        self.grid_size = grid_size
        self.settings = settings or DIFFICULTY_SETTINGS["hard"]
        self.stateless = (
            stateless  # If True, no EMA/accumulation - just per-frame processing
        )

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

    def process_batch(self, neural_data: np.ndarray) -> dict | None:
        """Process a batch of neural data.

        Args:
            neural_data: (batch_size, n_channels) array

        Returns:
            dict with heatmap, centroid, presence, etc. or None if still initializing
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
            center_distance = compute_center_distance(centroid, self.grid_size)
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
            center_distance = compute_center_distance(centroid, self.grid_size)

        return {
            "heatmap": grid,
            "centroid": centroid,
            "center_distance": center_distance,
            "bad_channels": int(self.bad_mask.sum())
            if self.bad_mask is not None
            else 0,
        }
