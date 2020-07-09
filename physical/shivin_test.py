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
from perception import CameraIntrinsics, RgbdSensorFactory, Image
# from pixel_selector import PixelSelector
from autolab_core import Point, Box, RigidTransform, YamlConfig, Logger
# from ambidex.databases.postgres import YamlLoader, PostgresSchema
# from ambidex.class_registry import postgres_base_cls_map, full_cls_list
# from ambidex.envs.actions import Grasp3D
# from grasp_utils import *
# from image_utils import *
logger = Logger.get_logger('grasp.py')
import yumipy

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
    def __init__(self, robot, calib_dir):
        self.robot = robot
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
        self.workspace = None
        if 'workspace' in phoxi_config.keys():
            self.workspace = Box(np.array(phoxi_config['workspace']['min_pt']),
                            np.array(phoxi_config['workspace']['max_pt']),
                            frame='world')

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

    def capture_image(self, frame=0):
        logger.info('Capturing depth image %d' %(frame))
        color, depth, _ = self.sensor.frames()
        # depth = depth.inpaint()
        depth.save('ros_phoxi/depth_%d.png' %(frame))
        color.save('ros_phoxi/color_%d.png' %(frame))

        foreground = color.foreground_mask(120, use_hsv=False)
        fg = color.mask_binary(foreground)
        fg.save('ros_phoxi/foreground_hsv_%d.png' %(frame))

        segment = fg.segment_kmeans(1,3)
        seg1 = fg.mask_binary(segment.segment_mask(1))
        seg2 = fg.mask_binary(segment.segment_mask(2))
        # seg3 = fg.mask_binary(segment.segment_mask(3))
        seg1.save('ros_phoxi/segment_1.png')
        seg2.save('ros_phoxi/segment_2.png')
        # seg3.save('ros_phoxi/segment_3.png')


        # depth.save('ros_phoxi/depth_%d.png' %(frame))
        # # deproject into 3D world coordinates
        # point_cloud_cam = self.camera_intr.deproject(depth)
        # point_cloud_cam.remove_zero_points()
        # point_cloud_world = self.T_camera_world * point_cloud_cam
        # seg_point_cloud_world, _ = point_cloud_world.box_mask(self.workspace)
        
        # raw_depth_path = os.path.join(self.save_dir, 'raw_depth_%d.npy' %(frame))
        # np.save(raw_depth_path, depth.data)

        # # compute the segmask for points above the box
        # seg_point_cloud_cam = self.T_camera_world.inverse() * seg_point_cloud_world
        # depth_im_seg = self.camera_intr.project_to_image(seg_point_cloud_cam)
        # segmask = depth_im_seg.to_binary()

        # # rescale segmask
        # if self.rescale_factor != 1.0:
        #     segmask = segmask.resize(rescale_factor, interp='nearest')
        
        # # save segmask
        # depth_im_seg.save('ros_phoxi/segdepth_%d.png' %(frame))

        # segmask.save('ros_phoxi/segmask_%d.png' %(frame))
    
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
    phys_robot = yumipy.YuMiRobot()
    task = Task(phys_robot, 'calib/phoxi')
    task.capture_image()
    # task.move()
    # grasp, orientation, drop = task.plan_grasp_drop()
    # task.grasp(grasp, orientation, drop)
    task.finish()
