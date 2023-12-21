import numpy as np


class DifferentialDriveModel:
    def __init__(self, Qt: np.ndarray) -> None:
        self.Qt = Qt

    def noisefree_motion(self, odometry: dict, x: np.ndarray) -> np.ndarray:
        # noise-free motion
        return x + np.array(
            [
                odometry["t"] * np.cos(x[2] + odometry["r1"]),
                odometry["t"] * np.sin(x[2] + odometry["r1"]),
                odometry["r1"] + odometry["r2"],
            ]
        )

    def jacobian(self, odometry: dict, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [1, 0, -odometry["t"] * np.sin(x[2] + odometry["r1"])],
                [0, 1, odometry["t"] * np.cos(x[2] + odometry["r1"])],
                [0, 0, 1],
            ]
        )
