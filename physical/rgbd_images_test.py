import rospy
# import click
import logging
import argparse
import numpy as np
import os
import random
import cv2
import time
import json
from perception import CameraIntrinsics, Image, DepthImage
from autolab_core import Box, RigidTransform, YamlConfig, Logger
logger = Logger.get_logger('shivin_test.py')
import yumipy

from colorfilters import HSVFilter
import cv2
from phoxipy import ColorizedPhoXiSensor
from phoxipy.phoxi_sensor import PhoXiSensor
from tools.utils import *

class Task(BaseTask):
    def __init__(self, sensor=None):
        super(Task, self).__init__()
        self.init_camera(sensor)
        self.init_robot()
        self.init_workspace()
        # self.init_attributes(base_path)

        # self.save_dir = "physical/ros_phoxi"
        # if not os.path.exists(self.save_dir):
        #     os.mkdir(self.save_dir)

    def capture_image(self, frame=0):
        logger.info('Capturing depth ')
        img = self.sensor.read()
        color, depth = img.color, img.depth
        # depth = depth.inpaint()
        depth.save('physical/ros_phoxi/depth.png')
        color.save('physical/ros_phoxi/color.png')

        seg_point_cloud_world = depth_to_world_seg(depth, self.camera_intr, self.T_camera_world, self.workspace)
        
        depth_seg = world_to_image(seg_point_cloud_world, self.camera_intr, self.T_camera_world)
        segmask = depth_seg.to_binary()

        # # rescale segmask
        # if self.rescale_factor != 1.0:
        #     segmask = segmask.resize(rescale_factor, interp='nearest')
        color_seg = color.mask_binary(segmask)

        # save segdepth
        depth_seg.save('physical/ros_phoxi/depth_seg.png')
        color_seg.save('physical/ros_phoxi/color_seg.png')

        window = HSVFilter(color_seg.data)
        window.show()

        # color_mask = color_seg.foreground_mask(100, use_hsv=False)
        # color_seg2 = color_seg.mask_binary(color_mask, invert=True)
        # color_seg2.save('physical/ros_phoxi/color_seg2.png')
        # depth_seg2 = depth_seg.mask_binary(color_mask, invert=True)
        # depth_seg2.save('physical/ros_phoxi/depth_seg2.png')
        # color_mask = color_seg.segment_hsv(np.array([80,60,0]), np.array([120,255,255])) #blue-green pipe connector
        # color_mask = color_seg.segment_hsv(np.array([90,60,0]), np.array([140,255,255])) #purple clamp
        # color_mask = color_seg.segment_hsv(np.array([0,0,1]), np.array([140,255,70])) #black tube
        # color_mask = color_seg.segment_hsv(np.array([0,100,1]), np.array([25,255,255])) #brown rock climbing hold
        # color_mask = color_seg.segment_hsv(np.array([80,60,0]), np.array([120,255,255])) #blue barclamp
        # color_mask = color_seg.segment_hsv(np.array([0,0,115]), np.array([180,110,255])) #white handrail bracket
        color_mask = color_seg.segment_hsv(np.array([0,0,75]), np.array([180,75,255])) #silver ornamental handrail bracket

        color_seg2 = color_seg.mask_binary(color_mask, invert=False)
        color_seg2.save('physical/ros_phoxi/color_seg2.png')
        depth_seg2 = depth_seg.mask_binary(color_mask, invert=False)
        depth_seg2.save('physical/ros_phoxi/depth_seg2.png')
        ind = np.where(depth_seg2.data != 0)
        center_i, center_j = np.mean(ind[0]), np.mean(ind[1])
        height, width = np.max(ind[0])- np.min(ind[0]), np.max(ind[1])- np.min(ind[1])
        print(height, width, center_i, center_j)
        depth_seg3 = depth_seg2.crop(256, 256, center_i, center_j)
        depth_seg3.save('physical/ros_phoxi/depth_seg3.png')
        # depth_seg4 = cv2.resize(depth_seg3.data, (128,128), interpolation = cv2.INTER_AREA)
        # depth_seg4 = DepthImage(depth_seg4)
        depth_seg4 = depth_seg3.resize((128,128), 'nearest')
        depth_seg4.save('physical/ros_phoxi/depth_seg4.png')

        nonzero_px = np.where(depth_seg.data != 0.0)
        print(np.max(depth_seg.data), np.min(depth_seg.data[nonzero_px[0], nonzero_px[1]]))
    
    def finish(self):
        self.robot.stop() # Stop the robot
        self.sensor.stop() # Stop the phoxi

if __name__ == '__main__':
    task = Task()
    task.capture_image()
    # task.move()
    task.finish()
