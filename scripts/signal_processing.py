"""
Signal processing for neural data streams.

Provides pure functions (EMAState, create_ema_state, update_ema, ema_activity)
for computing per-channel activity from raw voltage data.
Used by backend.py to transform sample_batch messages into features messages.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EMAState:
    """
    Immutable state for exponential moving average computation.

    Attributes:
        a_fast: Fast EMA decay coefficient
        a_base: Base EMA decay coefficient
        p_fast: Fast power EMA per channel (n_ch,) float32
        p_base: Base power EMA per channel (n_ch,) float32
    """

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
    """
    Factory function to create initial EMA state.

    Args:
        n_ch: Number of channels (typically 1024 for 32x32 grid)
        fs: Sampling frequency in Hz
        tau_fast_s: Fast EMA time constant in seconds
        tau_base_s: Base EMA time constant in seconds

    Returns:
        Initial EMAState with zero power and computed decay coefficients
    """
    dt = 1.0 / fs
    return EMAState(
        a_fast=float(dt / tau_fast_s),
        a_base=float(dt / tau_base_s),
        p_fast=np.zeros((n_ch,), dtype=np.float32),
        p_base=np.zeros((n_ch,), dtype=np.float32),
    )


def update_ema(state: EMAState, batch: np.ndarray) -> EMAState:
    """
    Process batch of samples and return updated state (pure function).

    Args:
        state: Current EMAState
        batch: (B, n_ch) float32 array of neural data samples

    Returns:
        New EMAState with updated p_fast and p_base
    """
    p_fast = state.p_fast.copy()
    p_base = state.p_base.copy()
    for x in batch:
        x2 = x * x
        p_fast = (1.0 - state.a_fast) * p_fast + state.a_fast * x2
        p_base = (1.0 - state.a_base) * p_base + state.a_base * x2
    return EMAState(
        a_fast=state.a_fast,
        a_base=state.a_base,
        p_fast=p_fast,
        p_base=p_base,
    )


def ema_activity(state: EMAState) -> np.ndarray:
    """
    Compute activity level (RMS-ish) from state (pure read).

    Args:
        state: Current EMAState

    Returns:
        Per-channel activity array
    """
    return np.sqrt(state.p_fast + 1e-12)


def ema_normalized_activity(state: EMAState) -> np.ndarray:
    """
    Compute log-normalized activity (fast relative to baseline) (pure read).

    Args:
        state: Current EMAState

    Returns:
        Per-channel normalized activity array
    """
    return np.log(state.p_fast + 1e-12) - np.log(state.p_base + 1e-12)


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
