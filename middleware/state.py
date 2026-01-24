from dataclasses import dataclass

import numpy as np


@dataclass
class SharedState:
    fs: float = 500.0
    n_ch: int = 1024
    last_t: float = 0.0
    total_samples: int = 0
    connected_to_input: bool = False

    # Latest computed feature map (32x32)
    last_heatmap: np.ndarray | None = None
