import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from .position import Position

PAUSE_TIME = 0.01


def plot_state(position: Position, landmarks, map_limits, pause_time: float = 0.01):
    """Visualizes the state of the kalman filter.

    Displays the mean and standard deviation of the belief,
    the state covariance sigma and the position of the
    landmarks.

    landmark positions"""
    lx = []
    ly = []

    for i in range(len(landmarks)):
        lx.append(landmarks[i + 1][0])
        ly.append(landmarks[i + 1][1])

    # mean of belief as current estimate
    estimated_pose = position.mu

    # calculate and plot covariance ellipse
    covariance = position.sigma[0:2, 0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    # get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:, max_ind]
    max_eigval = eigenvals[max_ind]

    # get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:, min_ind]
    min_eigval = eigenvals[min_ind]

    # chi-square value for sigma confidence interval
    chisquare_scale = 2.2789

    # calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale * max_eigval)
    height = 2 * np.sqrt(chisquare_scale * min_eigval)
    angle = np.arctan2(max_eigvec[1], max_eigvec[0])

    # generate covariance ellipse
    ell = Ellipse(
        xy=(estimated_pose[0], estimated_pose[1]),
        width=width,
        height=height,
        angle=angle / np.pi * 180,
    )
    ell.set_alpha(0.25)

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, "bo", markersize=10)
    plt.quiver(
        estimated_pose[0],
        estimated_pose[1],
        np.cos(estimated_pose[2]),
        np.sin(estimated_pose[2]),
        angles="xy",
        scale_units="xy",
    )
    plt.axis(map_limits)

    plt.pause(pause_time)
