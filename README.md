KalmanPy
==================
<h3>Branch: "LouThreeState"</h3>

A Kalman Filter for 2nd order, 1D dynamics of a High Altitude Balloon

The code is configured for three states; position, velocity and acceleration with measurements on each and assumes process noise on velocity and acceleration, but not position. The code is fully matrix-based for easy extensibility.

This is adapted from the example code found at:

http://stackoverflow.com/questions/13901997/kalman-2d-filter-in-python

There are some issues with this code that have been addressed:

1. There's a bug in the original code where the "correct" or "update" phase of the Kalman algorithm precedes the "predict" phase.
2. The first estimate is produced by the Kalman algorithm. This is incorrect, Kalman processing can't start until the second time step. The first time step Kalman estimate must be initialized manually with a best guess.
3. Discrete equations are used instead of a full matrix-based solution.

