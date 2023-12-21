from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from src import ekf

PAUSE_TIME = 0.02
DATA_PATH = Path("data/")

MAP_LIMITS = (-1.0, 12.0, 0.0, 10.0)


def main():
    # implementation of an extended Kalman filter for robot pose estimation

    # plot preferences, interactive plotting mode
    _ = plt.figure()
    plt.axis(MAP_LIMITS)
    plt.ion()
    plt.show()

    print("Reading landmark positions")
    landmarks = ekf.data.read_world(DATA_PATH / "world.dat")

    print("Reading sensor data")
    sensor_readings = ekf.data.read_sensor_data(DATA_PATH / "sensor_data.dat")

    Qt = 2 * np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.02]])
    motion_model = ekf.motion.DifferentialDriveModel(Qt)

    observation_model = ekf.observation.RangeSensorModel(rt_variance=0.5)

    # initialize belief
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    position = ekf.position.Position(mu, sigma)

    # run kalman filter
    for timestep in range(len(sensor_readings) // 2):
        # plot the current state
        ekf.plot.plot_state(position, landmarks, MAP_LIMITS, PAUSE_TIME)

        # perform prediction step
        position = ekf.kalman.prediction_step(
            motion_model, sensor_readings[timestep, "odometry"], position
        )

        # perform correction step
        position = ekf.kalman.correction_step(
            observation_model, sensor_readings[timestep, "sensor"], position, landmarks
        )

    plt.show()


if __name__ == "__main__":
    main()
