# Extended Kalman Filter
Applied Extended Kalman Filter (EKF) algorithm to the mobile sensing problem.

Assuming a known map of the world, EKF allows to perform recursive state estimation: it aims at identifying the position and orientation of our mobile object in this world map.
EKF works in two steps: 
* the prediction step uses the previous step's position and odometer reading to estimate how much movement happened (in the form of a gaussian distribution).
* the correction step refines this estimate by taking into account observations from sensors (in this case)

## A few details about the problem

- **world**: 2D map with 9 landmarks spread across a rectangle layout.
- **motion model**: differential drive, using an odometer model $(x, y, \theta)$.
- **sensor data**: for each time step, our agent collects data about the estimated distance and bearings from a subset of the above-mentioned landmarks. (note: we only )
- **observation model**: range sensors only (not using bearings in the current implementation), showing the estimated distance from a subset of landmarks.


## Run
```python3 main.py```

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
