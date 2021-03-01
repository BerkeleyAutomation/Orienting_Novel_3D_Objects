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

class Task:
    def __init__(self, sensor=None):
        # INITIALIZE PHOXI CAMERA
        calib_dir = '/nfs/diskstation/calib/phoxi'
        phoxi_config = YamlConfig("physical/cfg/tools/colorized_phoxi.yaml")
        self.sensor_name = 'phoxi'
        self.sensor_config = phoxi_config['sensors'][self.sensor_name]

        self.sensor_type = self.sensor_config['type']
        self.sensor_frame = self.sensor_config['frame']
        if sensor:
            self.sensor = sensor
        else:
            self.sensor = ColorizedPhoXiSensor("1703005", 3, calib_dir='/nfs/diskstation/calib/', inpaint=False) 
            self.sensor.start()

        # logger.info('Ready to capture images from sensor %s' %(self.sensor_name))
        basedir = "/home/shivin/catkin_ws/src/ambidex/tests/cfg/"
        yaml_obj_loader = YamlObjLoader(basedir)

        self.robot = yaml_obj_loader('physical_yumi_no_jaw')
        self.home_pose = self.robot.right.get_pose()
        self.camera_intr = CameraIntrinsics.load(os.path.join(calib_dir, 'phoxi.intr'))
        self.T_camera_world = RigidTransform.load(os.path.join(calib_dir, 'phoxi_to_world.tf'))

        # read workspace bounds, this is necessary to segment out the point cloud that is on top of the bin/surface
        self.workspace = None
        if 'workspace' in phoxi_config.keys():
            self.workspace = Box(np.array(phoxi_config['plane_workspace']['min_pt']),
                            np.array(phoxi_config['plane_workspace']['max_pt']),
                            frame='world')

        # Orienting Objects: Have box around gripper
        x,y,z = self.home_pose.translation
        self.box_workspace = Box(np.array([x-0.075, y-0.075,z-0.05]),
                            np.array([x+0.075,y+0.075, z+0.15]), frame = "world")

        Logger.reconfigure_root()

        self.save_dir = "physical/ros_phoxi"
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        logger.info('Ready to capture images from sensor %s' %(self.sensor_name))

    # def Kate_start(self):
    #     """ Connect to the robot and reset to the home pose. """
    #     # iteratively attempt to initialize the robot
    #     initialized = False
    #     self.robot = None
    #     while not initialized:
    #         try:
    #             # open robot
    #             self.robot = YuMiRobot(debug=self.config['debug'],
    #                                    arm_type=self.config['arm_type'])                # reset the arm poses
    #             self.robot.set_z(self.zoning)
    #             self.robot.set_v(self.velocity)
    #             if self._reset_on_start:
    #                 self.robot.reset_bin()                # reset the tools
    #             self.parallel_jaw_tool = ParallelJawYuMiTool(self.robot,
    #                                                          self.T_robot_world,
    #                                                          self._parallel_jaw_config,
    #                                                          debug=self.config['debug'])
    #             self.suction_tool = SuctionYuMiTool(self.robot,
    #                                                 self.T_robot_world,
    #                                                 self._suction_config,
    #                                                 debug=self.config['debug'])
    #             self.right_push_tool = PushYuMiTool(self.robot,
    #                                                 self.T_robot_world,
    #                                                 self._right_push_config,
    #                                                 debug=self.config['debug'])
    #             self.left_push_tool = PushYuMiTool(self.robot,
    #                                                self.T_robot_world,
    #                                                self._left_push_config,
    #                                                debug=self.config['debug'])
    #             self.parallel_jaw_tool.open_gripper()
    #             self.suction_tool.open_gripper()                # mark initialized
    #             initialized = True

    def capture_image(self, frame=0):
        logger.info('Capturing depth ie %d')
        img = self.sensor.read()
        color, depth = img.color, img.depth
        # depth = depth.inpaint()
        depth.save('physical/ros_phoxi/depth.png')
        color.save('physical/ros_phoxi/color.png')

        seg_point_cloud_world = depth_to_world_seg(depth, self.camera_intr, self.T_camera_world, self.box_workspace)
        
        depth_seg = world_to_image(seg_point_cloud_world, self.camera_intr, self.T_camera_world)
        segmask = depth_seg.to_binary()

        # # rescale segmask
        # if self.rescale_factor != 1.0:
        #     segmask = segmask.resize(rescale_factor, interp='nearest')
        color_seg = color.mask_binary(segmask)

        # save segdepth
        depth_seg.save('physical/ros_phoxi/depth_seg.png')
        color_seg.save('physical/ros_phoxi/color_seg.png')

        # window = HSVFilter(color_seg.data)
        # window.show()

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
        color_mask = color_seg.segment_hsv(np.array([0,0,100]), np.array([180,120,255])) #gold handrail bracket

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
    
    def move(self):
        """params: all in frame of object --> world
        Computes all params in gripper to world frame, then executes a single action 
        of the robot picking up the rope at grasp_coord with orientation grasp_theta"""
        pose = self.robot.left.get_pose() # getting the current pose of the left end effector
        print(pose)
        y.left.goto_pose_delta((0,0,0), (5,-5,-10))

    def finish(self):
        self.robot.stop() # Stop the robot
        self.sensor.stop() # Stop the phoxi

if __name__ == '__main__':
    task = Task()
    task.capture_image()
    # task.move()
    task.finish()
