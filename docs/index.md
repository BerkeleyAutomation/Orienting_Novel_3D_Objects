Paper published and presented in [IEEE CASE 2020](https://www.imse.hku.hk/case2020/)

Orienting objects is a critical component in the automation of many packing and assembly tasks. We present an algorithm to orient novel objects given a depth image of the object in its current and desired orientation. We formulate a self-supervised objective for this problem and train a deep neural network to estimate the 3D rotation as parameterized by a quaternion, between these current and desired depth images. We then use the trained network in a proportional controller to re-orient objects based on the estimated rotation between the two depth images. Results suggest that in simulation we can rotate unseen objects with unknown geometries by up to 30° with a median angle error of 1.47° over 100 random initial/desired orientations each for 22 novel objects. Experiments on physical objects suggest that the controller can achieve a median angle error of 4.2° over 10 random initial/desired orientations each for 5 objects.

## Video

[![Orienting Novel 3D Objects Using Self-Supervised Learning of Rotation Transforms](https://img.youtube.com/vi/nij_JgNP1qw/0.jpg)](https://www.youtube.com/watch?v=nij_JgNP1qw "CASE 2020: Orienting Novel 3D Objects Using Self-Supervised Learning of Rotation Transforms")
