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
import autolab_core.utils as utils
from perception import CameraIntrinsics, RgbdSensorFactory, Image, DepthImage
# from pixel_selector import PixelSelector
from autolab_core import Point, Box, RigidTransform, YamlConfig, Logger
# from ambidex.databases.postgres import YamlLoader, PostgresSchema
# from ambidex.class_registry import postgres_base_cls_map, full_cls_list
# from ambidex.envs.actions import Grasp3D
# from grasp_utils import *
# from image_utils import *
logger = Logger.get_logger('shivin_test.py')
import yumipy
import cv2

# This is just a utility for loading an object from a config file
class YamlObjLoader(object):
    def __init__(self, basedir):
        self.basedir = basedir
        self._map = {}
        for root, dirs, fns in os.walk(basedir):
            for fn in fns:
                full_fn = os.path.join(root, fn)
                _, f = os.path.split(full_fn)
                if f in self._map:
                    raise ValueError('Duplicate file named {}'.format(f))
                self._map[f] = full_fn
        self._yaml_loader = YamlLoader(PostgresSchema('pg_schema', postgres_base_cls_map, full_cls_list))

    def load(self, key):
        key = key + '.yaml'
        full_filepath = self._map[key]
        return self._yaml_loader.load(full_filepath)

    def clear(self):
        self._yaml_loader.clear()

    def __call__(self, key):
        return self.load(key)

class Task:
    def __init__(self, calib_dir):
        self.robot = yumipy.YuMiRobot(include_right=False)
        # self.gripper = self.robot.grippers[0] # This is the Parallel Jaw of the YuMi
        self.gripper_home_pose = self.robot.left.get_pose()
        self.camera_intr = CameraIntrinsics.load(os.path.join(calib_dir, 'phoxi.intr'))
        self.T_camera_world = RigidTransform.load(os.path.join(calib_dir, 'phoxi_to_world.tf'))

        # self.pixel_selector = PixelSelector() # A utility to select pixels from an image

        self.image = None
        self.depth = None

        # INITIALIZE PHOXI CAMERA
        phoxi_config = YamlConfig("cfg/tools/colorized_phoxi.yaml")
        # read rescale factor
        self.rescale_factor = 1.0
        if 'rescale_factor' in phoxi_config.keys():
            self.rescale_factor = config['rescale_factor']

        # read workspace bounds, this is necessary to segment out the point cloud that is on top of the bin/surface
        # self.workspace = None
        # if 'workspace' in phoxi_config.keys():
        #     self.workspace = Box(np.array(phoxi_config['workspace']['min_pt']),
        #                     np.array(phoxi_config['workspace']['max_pt']),
        #                     frame='world')
        x,y,z = self.gripper_home_pose.translation
        self.workspace = Box(np.array([x-0.075, y-0.075,z]),
                            np.array([x+0.075,y+0.075, z+0.2]), frame = "world")

        rospy.init_node('capture_test_images') #NOTE: this is required by the camera sensor classes
        Logger.reconfigure_root()

        self.sensor_name = 'phoxi'
        self.sensor_config = phoxi_config['sensors'][self.sensor_name]
        # self.demo_horizon_length = phoxi_config['sensors'][self.sensor_name]['num_images']

        logger.info('Ready to capture images from sensor %s' %(self.sensor_name))
        self.save_dir = "ros_phoxi"
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)        
        
        # read params
        self.sensor_type = self.sensor_config['type']
        self.sensor_frame = self.sensor_config['frame']
        self.sensor = RgbdSensorFactory.sensor(self.sensor_type, self.sensor_config)
        self.sensor.start()

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
        logger.info('Capturing depth image %d' %(frame))
        color, depth, _ = self.sensor.frames()
        # depth = depth.inpaint()
        depth.save('ros_phoxi/depth_%d.png' %(frame))
        color.save('ros_phoxi/color_%d.png' %(frame))

        # deproject into 3D world coordinates
        point_cloud_cam = self.camera_intr.deproject(depth)
        point_cloud_cam.remove_zero_points()
        point_cloud_world = self.T_camera_world * point_cloud_cam
        seg_point_cloud_world, _ = point_cloud_world.box_mask(self.workspace)
        
        # compute the segmask for points above the box
        seg_point_cloud_cam = self.T_camera_world.inverse() * seg_point_cloud_world
        depth_seg = self.camera_intr.project_to_image(seg_point_cloud_cam)
        segmask = depth_seg.to_binary()
        # # rescale segmask
        # if self.rescale_factor != 1.0:
        #     segmask = segmask.resize(rescale_factor, interp='nearest')
        color_seg = color.mask_binary(segmask)

        # save segdepth
        depth_seg.save('ros_phoxi/depth_seg_%d.png' %(frame))
        color_seg.save('ros_phoxi/color_seg_%d.png' %(frame))

        # segment = color_seg.segment_kmeans(1,3)
        # seg1 = color_seg.mask_binary(segment.segment_mask(1))
        # seg2 = color_seg.mask_binary(segment.segment_mask(2))
        # seg1.save('ros_phoxi/segment_1.png')
        # seg2.save('ros_phoxi/segment_2.png')
        color_mask = color_seg.foreground_mask(100, use_hsv=False)
        color_seg2 = color_seg.mask_binary(color_mask, invert=True)
        color_seg2.save('ros_phoxi/color_seg2_%d.png' %(frame))
        depth_seg2 = depth_seg.mask_binary(color_mask, invert=True)
        depth_seg2.save('ros_phoxi/depth_seg2_%d.png' %(frame))
        ind = np.where(depth_seg2.data != 0)
        center_i, center_j = np.mean(ind[0]), np.mean(ind[1])
        height, width = np.max(ind[0])- np.min(ind[0]), np.max(ind[1])- np.min(ind[1])
        print(height, width, center_i, center_j)
        depth_seg3 = depth_seg2.crop(256, 256, center_i, center_j)
        depth_seg3.save('ros_phoxi/depth_seg3_%d.png' %(frame))
        # depth_seg4 = cv2.resize(depth_seg3.data, (128,128), interpolation = cv2.INTER_AREA)
        # depth_seg4 = DepthImage(depth_seg4)
        depth_seg4 = depth_seg3.resize((128,128), 'nearest')
        depth_seg4.save('ros_phoxi/depth_seg4_%d.png' %(frame))

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
    task = Task('/nfs/diskstation/calib/phoxi')
    task.capture_image()
    # task.move()
    # grasp, orientation, drop = task.plan_grasp_drop()
    # task.grasp(grasp, orientation, drop)
    task.finish()
