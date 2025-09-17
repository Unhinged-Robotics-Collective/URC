# Steps to generate trajectories

1. Capture hand landmarks into xyz data DONE
2. Visualize hand landmarks DONE
3. Create hand frame with direction vectors, etc DONE
4. One camera makes the hand be 2D, add second camera to record 3D DONE - the second camera can already record
5. Filter the hand positions to be smoother => temporal interpolation DONE
6. Derive some mathematical model to represent the inverse transformations for two cameras
6. Process data to move one single hand in 3 axes - preferably without ground truth
NOTES: https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
https://en.wikipedia.org/wiki/Perspective-n-Point
7. Record this data
