# Extended Kalman Filter
Applied Extended Kalman Filter (EKF) algorithm to the mobile sensing problem.

Assuming a known map of the world, EKF allows to perform recursive state estimation: it aims at identifying the position and orientation of our mobile object in this world map.
EKF works in two steps: 
* the prediction step uses the previous step's position and odometer reading to estimate how much movement happened (in the form of a gaussian distribution).
* the correction step refines this estimate by taking into account observations from sensors (in this case)

## The Setup

- **world**: 2D map with 9 landmarks spread across a rectangle layout.
- **motion model**: differential drive, using an odometer model $(x, y, \theta)$.
- **sensor data**: for each time step, our agent collects data about the estimated distance and bearings from a subset of the above-mentioned landmarks. (note: we only take into account the range data so far, and discard bearings information.)
- **observation model**: range sensors only (not using bearings in the current implementation), showing the estimated distance from a subset of landmarks.  


## Program Output 
<img src="imgs/example.png" alt="drawing" width="500"/>
A dynamic visualization of the estimated state after each time step.

## Run
```python3 main.py``` displays a visualization of the word and the predicted object position step by step given odometry and range sensor readings.

## Observations
the program easily allows to evaluate the impact of various factors such as:
* removing the correction step: the mean position estimate remains relatively close, but uncertainty around that position explodes over time.
* removing the prediction step: the position estimate is significantly degraded, but the model is over-confident about its position estimate.
* reducing the number of observations per time step to only 1 landmark: negatively impacts both the mean position estimate, and the variance for that estimate.
* increasing uncertainty on range (e.g. by moving to a sonar)

## Contributions vs original exercise
- vectorization of the implementation to improve performance
- refactored code to decouple motion and observation models from EKF algorithm core implementation.
    - the code can therefore easily be extended to other motion / observation models. all you need is to implement the [`ObservationModel`](src/ekf/kalman.py) and/or [`MotionModel`](src/ekf/kalman.py).

## Opportunities for improvement

- [x] decouple models from EKF core implementation
- [ ] move to pytorch for higher efficiency
- [ ] turn into CLI allowing to control input data
- [ ] integrate bearings information to model


## Outstanding questions
- in this exercise, $Q_t$ and $R_t$ are given. how would they be estimated in practice?


## Acknowledgements
http://ais.informatik.uni-freiburg.de/teaching/ss23/robotics/exercises/sheet07.pdf
