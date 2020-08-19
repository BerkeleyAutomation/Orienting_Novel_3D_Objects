Paper published in IEEE CASE 2020. Available on [arXiv](todo_publish)

Orienting objects is a critical component in theautomation of many packing and assembly tasks. We presentan algorithm to orient novel objects given a depth image ofthe object in its current and desired orientation. We formulatea self-supervised objective for this problem and train a deepneural network to estimate the 3D rotation as parameterized bya quaternion, between these current and desired depth images.We then use the trained network in a proportional controller tore-orient objects based on the estimated rotation between thetwo depth images. Results suggest that in simulation we can ro-tate unseen objects with unknown geometries by up to 30° witha median angle error of 1.47° over 100 random initial/desiredorientations each for 22 novel objects. Experiments on physicalobjects suggest that the controller can achieve a median angleerror of 4.2° over 10 random initial/desired orientations eachfor 5 objects.

## Video

[![Orienting Novel 3D Objects Using Self-Supervised Learning of Rotation Transforms](https://img.youtube.com/vi/nij_JgNP1qw/0.jpg)](https://www.youtube.com/watch?v=nij_JgNP1qw "CASE 2020: Orienting Novel 3D Objects Using Self-Supervised Learning of Rotation Transforms")
