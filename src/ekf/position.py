from dataclasses import dataclass

import numpy as np


@dataclass
class Position:
    mu: np.ndarray
    sigma: np.ndarray
