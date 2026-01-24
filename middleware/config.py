from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    input_url: str = "ws://localhost:8765"
    out_host: str = "localhost"
    out_port: int = 8787

    # Signal processing params
    band_hz: tuple[float, float] = (70.0, 150.0)
    filter_order: int = 2
    tau_fast_s: float = 0.25
    tau_base_s: float = 8.0

    # Spatial smoothing
    spatial_sigma: float = 1.0  # set to 0.0 if you want max stability

    # Output rate to frontend
    out_hz: float = 20.0

    # WebSocket stability
    max_queue: int = 2
    ping_interval: float | None = None  # disable pings
    ping_timeout: float | None = None   # disable timeout
    close_timeout: float = 5.0
