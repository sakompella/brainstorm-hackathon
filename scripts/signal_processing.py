"""
Signal processing for neural data streams.

Provides ActivityEMA for computing per-channel activity from raw voltage data.
Used by backend.py to transform sample_batch messages into features messages.
"""

import numpy as np


class ActivityEMA:
    """
    Maintains a fast and slow EMA of signal power (x^2) per channel.
    Output = sqrt(fast_power) (an RMS-ish activity level).

    Args:
        n_ch: Number of channels (typically 1024 for 32x32 grid)
        fs: Sampling frequency in Hz
        tau_fast_s: Fast EMA time constant in seconds (responsive smoothing)
        tau_base_s: Base EMA time constant in seconds (slow baseline)
    """

    def __init__(
        self,
        n_ch: int,
        fs: float,
        tau_fast_s: float = 0.2,
        tau_base_s: float = 8.0,
    ):
        dt = 1.0 / fs
        self.a_fast = float(dt / tau_fast_s)
        self.a_base = float(dt / tau_base_s)
        self.p_fast = np.zeros((n_ch,), dtype=np.float32)
        self.p_base = np.zeros((n_ch,), dtype=np.float32)

    def update_batch(self, batch: np.ndarray) -> None:
        """
        Update EMA with batch of samples.

        Args:
            batch: (B, n_ch) float32 array of neural data samples
        """
        for x in batch:
            x2 = x * x
            self.p_fast = (1.0 - self.a_fast) * self.p_fast + self.a_fast * x2
            self.p_base = (1.0 - self.a_base) * self.p_base + self.a_base * x2

    def activity(self) -> np.ndarray:
        """Return current activity level (RMS-ish) per channel."""
        return np.sqrt(self.p_fast + 1e-12)

    def normalized_activity(self) -> np.ndarray:
        """Return log-normalized activity (fast relative to baseline)."""
        return np.log(self.p_fast + 1e-12) - np.log(self.p_base + 1e-12)


def compute_presence(activity: np.ndarray, k: int = 50) -> float:
    """
    Compute global presence indicator from activity.

    Args:
        activity: Per-channel activity array
        k: Number of top channels to average

    Returns:
        Mean activity of top-k channels
    """
    topk = np.partition(activity, -k)[-k:]
    return float(np.mean(topk))
