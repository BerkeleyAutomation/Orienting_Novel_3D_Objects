# import rospy
# import click
# import logging
# import argparse
# import numpy as np
# import os
# import random
# import cv2
# import time
# import json
# import autolab_core.utils as utils
# from perception import CameraIntrinsics, RgbdSensorFactory, Image
# from pixel_selector import PixelSelector
# from autolab_core import Point, Box, RigidTransform, YamlConfig, Logger
# from ambidex.databases.postgres import YamlLoader, PostgresSchema
# from ambidex.class_registry import postgres_base_cls_map, full_cls_list
# from ambidex.envs.actions import Grasp3D
# from grasp_utils import *
# from image_utils import *
# logger = Logger.get_logger('grasp.py')

# # This is just a utility for loading an object from a config file
# class YamlObjLoader(object):
# def __init__(self, basedir):
# self.basedir = basedir
# self._map = {}
# for root, dirs, fns in os.walk(basedir):
# for fn in fns:
# full_fn = os.path.join(root, fn)
# _, f = os.path.split(full_fn)
# if f in self._map:
# raise ValueError('Duplicate file named {}'.format(f))
# self._map[f] = full_fn
# self._yaml_loader = YamlLoader(PostgresSchema('pg_schema', postgres_base_cls_map, full_cls_list))

# def load(self, key):
# key = key + '.yaml'
# full_filepath = self._map[key]
# return self._yaml_loader.load(full_filepath)

# def clear(self):
# self._yaml_loader.clear()

# def __call__(self, key):
# return self.load(key)

# class Task:
# # Take a photo of the workspace, select a point (pixel) to grasp, and grasp the point

# def __init__(self, robot, calib_dir, base_dir='/home/priya/catkin_ws/src/rope_manip', phoxi_config_filename='cfg/tools/capture_test_images.yaml'):
# self.robot = robot
# self.gripper = self.robot.grippers[0] # This is the Parallel Jaw of the YuMi
# self.gripper_home_pose = self.robot.left.get_pose()
# self.camera_intr = CameraIntrinsics.load(os.path.join(calib_dir, 'phoxi.intr'))
# self.T_camera_world = RigidTransform.load(os.path.join(calib_dir, 'phoxi_to_world.tf'))

# self.pixel_selector = PixelSelector() # A utility to select pixels from an image

# self.image = None
# self.depth = None

# # INITIALIZE PHOXI CAMERA
# phoxi_config = YamlConfig(os.path.join(base_dir, phoxi_config_filename))
# # read rescale factor
# self.rescale_factor = 1.0
# if 'rescale_factor' in phoxi_config.keys():
# self.rescale_factor = config['rescale_factor']

# # read workspace bounds, this is necessary to segment out the point cloud that is on top of the bin/surface
# self.workspace = None
# if 'workspace' in phoxi_config.keys():
# self.workspace = Box(np.array(phoxi_config['workspace']['min_pt']),
# np.array(phoxi_config['workspace']['max_pt']),
# frame='world')

# rospy.init_node('capture_test_images') #NOTE: this is required by the camera sensor classes
# Logger.reconfigure_root()

# self.sensor_name = 'phoxi'
# self.sensor_config = phoxi_config['sensors'][self.sensor_name]
# self.demo_horizon_length = phoxi_config['sensors'][self.sensor_name]['num_images']

# logger.info('Ready to capture images from sensor %s' %(self.sensor_name))
# self.base_dir = base_dir
# self.save_dir = os.path.join(base_dir, 'grasping_images')
# if not os.path.exists(self.save_dir):
# os.mkdir(self.save_dir)
# # read params
# self.sensor_type = self.sensor_config['type']
# self.sensor_frame = self.sensor_config['frame']
# self.sensor = RgbdSensorFactory.sensor(self.sensor_type, self.sensor_config)
# self.sensor.start()

# def capture_image(self, frame=0):
# logger.info('Capturing image %d' %(frame))
# color, depth, ir = self.sensor.frames()

# # save processed images
# if self.workspace is not None:
# # deproject into 3D world coordinates
# point_cloud_cam = self.camera_intr.deproject(depth)
# point_cloud_cam.remove_zero_points()
# point_cloud_world = self.T_camera_world * point_cloud_cam
# seg_point_cloud_world, _ = point_cloud_world.box_mask(self.workspace)
# raw_depth_path = os.path.join(self.save_dir, 'raw_depth_%d.npy' %(frame))
# np.save(os.path.join(self.save_dir, 'raw_depth_%d.npy' %(frame)), depth.data)

# # compute the segmask for points above the box
# seg_point_cloud_cam = self.T_camera_world.inverse() * seg_point_cloud_world
# depth_im_seg = self.camera_intr.project_to_image(seg_point_cloud_cam)
# segmask = depth_im_seg.to_binary()

# # rescale segmask
# if self.rescale_factor != 1.0:
# segmask = segmask.resize(rescale_factor, interp='nearest')
# # save segmask
# depth_im_seg.save(os.path.join(self.save_dir, 'segdepth_%d.png' %(frame)))

# segmask.save(os.path.join(self.save_dir, 'segmask_%d.png' %(frame)))
# def move(self):
# """params: all in frame of object --> world
# Computes all params in gripper to world frame, then executes a single action
# of the robot picking up the rope at grasp_coord with orientation grasp_theta"""
# pose = self.robot.left.get_pose() # getting the current pose of the left end effector
# print(pose)
# # Tra: [0.40452 0.46798003 0.55596006]
# # Rot: [[-0.99960609 -0.00183157 -0.02800547]
# # [-0.00216749 0.99992602 0.01196921]
# # [ 0.02798148 0.0120252 -0.99953611]]
# # Qtn: [-0.01399837 -0.00099988 0.99988352 0.0059993 ]

# # pose.translation[2] -= 0.01
# pose.translation[0] = 0.532
# pose.translation[1] = 0.071
# pose.translation[2] = 0.241
# pose.quaternion[0] = 0.19095
# pose.quaternion[0] = -0.95823
# pose.quaternion[0] = -0.20912
# pose.quaternion[0] = 0.03997
# self.robot.left.goto_pose(pose) # move left arm by 5cm using goto_pose

# # self.robot.left.goto_pose_delta((0.0,0,-0.01)) # move left arm by 5cm using move delta

# def grasp(self, grasp_coord, grasp_theta, drop_coord):
# """params: all in frame of object --> world
# Computes all params in gripper to world frame, then executes a single action
# of the robot picking up the rope at grasp_coord with orientation grasp_theta"""
# obj_pose = RigidTransform(translation=np.array(grasp_coord), from_frame="object", to_frame="world")
# grasp_coord_gripper = self.gripper.center(pose=obj_pose)

# # Adding stuff for drop coord
# obj_drop_pose = RigidTransform(translation=np.array(drop_coord), from_frame="object", to_frame="world")
# drop_coord_gripper = self.gripper.center(pose=obj_drop_pose)
# grasp_orientation = np.array([[np.cos(grasp_theta), -np.sin(grasp_theta), 0], [np.sin(grasp_theta), np.cos(grasp_theta), 0], [0, 0, 1]])
# pca_grasp_pose = RigidTransform(rotation=grasp_orientation, from_frame="world", to_frame="world")*self.gripper_home_pose
# grasp_pose_gripper = RigidTransform(rotation=pca_grasp_pose.rotation, translation=np.array(grasp_coord_gripper), from_frame="gripper", to_frame="world")
# drop_pose_gripper = RigidTransform(rotation=pca_grasp_pose.rotation, translation=np.array(drop_coord_gripper), from_frame="gripper", to_frame="world")

# #grasp = Grasp3D(self.gripper, grasp_pose_gripper, drop_pose=None)
# grasp = Grasp3D(self.gripper, grasp_pose_gripper, drop_pose=drop_pose_gripper)

# tool = self.robot.select_tool(grasp)
# T_pregrasp_world, T_approach_world, T_grasp_world, T_lift_world, avoid_singularities = tool.plan(grasp)
# print "APPROACH", T_approach_world.translation
# print "GRASP", T_grasp_world.translation
# print "LIFT", T_lift_world.translation
# if click.confirm('Do you want to continue?', default=True):
# print('Executing action...')
# self.robot.execute(grasp)
# else:
# if click.confirm('Do you want to quit?', default=False):
# self.finish()

# def finish(self):
# self.robot.stop() # Stop the robot
# self.sensor.stop() # Stop the phoxi

# if __name__ == '__main__':
# basedir = os.path.join(os.path.dirname(__file__), '..', '..', 'ambidex', 'tests', 'cfg')
# yaml_obj_loader = YamlObjLoader(basedir)
# phys_robot = yaml_obj_loader('physical_yumi')
# task = Task(phys_robot, '/home/priya/catkin_ws/src/rope_manip/calib/phoxi')
# # task.capture_image()
# task.move()
# # grasp, orientation, drop = task.plan_grasp_drop()
# # task.grasp(grasp, orientation, drop)
# task.finish()
