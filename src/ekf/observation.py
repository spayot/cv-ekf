from typing import Protocol

import numpy as np


class RangeSensorModel:
    def __init__(self, rt_variance: float = 0.5) -> None:
        self.rt_variance = rt_variance

    def expected_distance(self, x, observed_landmarks) -> np.ndarray:
        return np.linalg.norm(x[:2] - observed_landmarks, axis=1)

    def jacobian(self, mu, observed_landmarks) -> np.ndarray:
        Ht = np.array(
            [
                mu[0] - observed_landmarks[:, 0],
                mu[1] - observed_landmarks[:, 0],
                np.zeros(len(observed_landmarks)),
            ]
        ).T
        return Ht / self.expected_distance(mu, observed_landmarks).reshape(-1, 1)

    def Rt(self, observed_landmarks) -> np.ndarray:
        return self.rt_variance * np.eye(len(observed_landmarks))
