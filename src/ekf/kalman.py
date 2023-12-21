from typing import Protocol

import numpy as np

from .position import Position


class MotionModel(Protocol):
    Qt: np.ndarray

    def noisefree_motion(self, odometry: dict[str, float], x: np.ndarray) -> np.ndarray:
        ...

    def jacobian(self, odometry: dict[str, float], x: np.ndarray) -> np.ndarray:
        ...


class ObservationModel(Protocol):
    def expected_distance(
        self, x: np.ndarray, observed_landmarks: np.ndarray
    ) -> np.ndarray:
        ...

    def jacobian(self, mu, observed_landmarks) -> np.ndarray:
        ...

    def Rt(self, observed_landmarks) -> np.ndarray:
        ...


def prediction_step(
    model: MotionModel, odometry: dict[str, float], position: Position
) -> Position:
    new_mu = model.noisefree_motion(odometry, position.mu)

    Gt = model.jacobian(odometry, position.mu)
    new_sigma = Gt.dot(position.sigma).dot(Gt.T) + model.Qt

    return Position(new_mu, new_sigma)


def kalman_gain(
    model: ObservationModel,
    position: Position,
    observed_landmarks: np.ndarray,
):
    Ht = model.jacobian(position.mu, observed_landmarks)
    Kt = np.linalg.inv(Ht.dot(position.sigma).dot(Ht.T) + model.Rt(observed_landmarks))
    return position.sigma.dot(Ht.T).dot(Kt)


def correction_step(
    model: ObservationModel,
    sensor_data: dict,
    position: Position,
    landmarks: dict[int, np.ndarray],
) -> Position:
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    #
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution

    # measured landmark ids and ranges
    ids = sensor_data["id"]
    ranges = sensor_data["range"]

    observed_landmarks = np.array([landmarks[idx] for idx in ids])

    Ht = model.jacobian(position.mu, observed_landmarks)
    Kt = kalman_gain(model, position, observed_landmarks)

    mu = position.mu + Kt.dot(
        np.array(ranges) - model.expected_distance(position.mu, observed_landmarks)
    )
    sigma = position.sigma - Kt.dot(Ht).dot(position.sigma)

    return Position(mu, sigma)
