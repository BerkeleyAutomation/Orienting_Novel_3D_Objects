import numpy as np
import cv2
import os

#z_near = 0.6
#z_far = 0.85
z_near = 0.7
z_far = 0.8

filelist = os.listdir('.')
for f in filelist[:]: # filelist[:] makes a copy of filelist.
    if f.endswith(".exr"):
        a = cv2.imread(f, -cv2.IMREAD_ANYDEPTH)
        a = (a - a.min()) / (a.max() - a.min()) * (z_far - z_near) + z_near
        print(a.max())
        print(a.min())
        a.astype(np.uint16)
        cv2.imwrite("scaled_" + f, a)

