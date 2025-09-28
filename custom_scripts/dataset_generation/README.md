# Steps to generate trajectories

1. Capture hand landmarks into xyz data DONE
2. Visualize hand landmarks DONE
3. Create hand frame with direction vectors, etc DONE
4. One camera makes the hand be 2D, add second camera to record 3D DONE - the second camera can already record
5. Filter the hand positions to be smoother => temporal interpolation DONE
6. Derive some mathematical model to represent the inverse transformations for two cameras - DONE
7. Inherit Z coordinate from one camera. If the cameras are not orthogonal, then it's not exactly just one axis. Use some freeki maths to derive the z distance.
- use homogenous coordinates: p_1 = \lambda \cdot [u,v,1] => the lambda is probably the distance along the z axis of the camera
- i know the point correspondence between camera, so I just wanna find the line intersection, lambda_1 & lambda_2
    - => \lambda_1 \cdot p_1 = R @ \lambda_2 \cdot p_2 + \vec{t} => LSQ of this
6. Process data to move one single hand in 3 axes - preferably without ground truth
NOTES: https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
https://en.wikipedia.org/wiki/Perspective-n-Point
7. Record this data
