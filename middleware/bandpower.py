import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class BandpowerEMA:
    """
    Streaming-safe bandpower:
      raw -> causal bandpass (sosfilt with state) -> power (x^2) -> EMA fast/base
    """

    def __init__(
        self,
        n_ch: int,
        fs: float,
        band_hz: tuple[float, float] = (70.0, 150.0),
        filter_order: int = 4,
        tau_fast_s: float = 0.25,
        tau_base_s: float = 8.0,
    ):
        self.n_ch = n_ch
        self.fs = fs
        self.band_hz = band_hz

        low, high = band_hz
        self.sos = butter(
            filter_order,
            [low, high],
            btype="bandpass",
            fs=fs,
            output="sos",
        )

        # SciPy expects zi shape (n_sections, 2, n_ch) for axis=0 with (B, n_ch)
        zi0 = sosfilt_zi(self.sos).astype(np.float32)  # (n_sections, 2)
        self.zi = np.repeat(zi0[:, :, None], n_ch, axis=2)  # (n_sections, 2, n_ch)

        dt = 1.0 / fs
        self.a_fast = float(dt / tau_fast_s)
        self.a_base = float(dt / tau_base_s)

        self.p_fast = np.zeros((n_ch,), dtype=np.float32)
        self.p_base = np.zeros((n_ch,), dtype=np.float32)

    def update_batch(self, batch_raw: np.ndarray) -> None:
        y, self.zi = sosfilt(self.sos, batch_raw, axis=0, zi=self.zi)  # (B, n_ch)
        y2 = y * y

        for x2 in y2:
            self.p_fast = (1.0 - self.a_fast) * self.p_fast + self.a_fast * x2
            self.p_base = (1.0 - self.a_base) * self.p_base + self.a_base * x2

    def normalized_vec(self) -> np.ndarray:
        return (np.log(self.p_fast + 1e-12) - np.log(self.p_base + 1e-12)).astype(np.float32)
    
    def normalized_vec(self, use_percentile=True):
        vec = (np.log(self.p_fast + 1e-12) - np.log(self.p_base + 1e-12))
        
        if use_percentile:
            # <1ms - negligible latency
            p5 = np.percentile(vec, 5)
            p95 = np.percentile(vec, 95)
            vec = (vec - p5) / (p95 - p5 + 1e-9)
        
        return vec.astype(np.float32)




#create a multi-band version
class MultiBandPowerEMA:
    """Track power across multiple frequency bands"""
    def __init__(self, n_ch, fs, bands, tau_fast_s=0.25, tau_base_s=8.0):
        self.bands = bands  # e.g., {'highgamma': (70,150), 'beta': (12,30)}
        self.estimators = {
            name: BandpowerEMA(n_ch, fs, band_hz=band, tau_fast_s=tau_fast_s, tau_base_s=tau_base_s)
            for name, band in bands.items()
        }
    
    def update_batch(self, batch_raw):
        for estimator in self.estimators.values():
            estimator.update_batch(batch_raw)
    
    def get_combined_heatmap(self, weights=None):
        """Combine multiple bands with optional weights"""
        if weights is None:
            weights = {name: 1.0 for name in self.bands.keys()}
        
        combined = np.zeros_like(next(iter(self.estimators.values())).normalized_vec())
        total_weight = sum(weights.values())
        
        for name, estimator in self.estimators.items():
            combined += weights.get(name, 1.0) * estimator.normalized_vec()
        
        return combined / total_weight