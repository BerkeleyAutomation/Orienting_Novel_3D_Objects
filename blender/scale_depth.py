import numpy as np
import cv2
import os

z_near = 0.5
z_far = 1.5

filelist = os.listdir('.')
for f in filelist[:]: # filelist[:] makes a copy of filelist.
    if f.endswith(".png"):
        a = cv2.imread(f, -1)
        a = (a - a.min()) / (a.max() - a.min()) * (z_far - z_near) + z_near
        print(a.max())
        print(a.min())
        cv2.imwrite("scaled_" + f, a)

