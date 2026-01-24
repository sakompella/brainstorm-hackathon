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
        zi0 = sosfilt_zi(self.sos).astype(np.float32)          # (n_sections, 2)
        self.zi = np.repeat(zi0[:, :, None], n_ch, axis=2)     # (n_sections, 2, n_ch)

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
