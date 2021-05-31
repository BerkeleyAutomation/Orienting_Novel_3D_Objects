# from __future__ import division
from autolab_core import YamlConfig, RigidTransform, TensorDataset, Box, Logger
from scipy.spatial.transform import Rotation
import os
import time
import sys

import numpy as np
import itertools
import argparse

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt

import random
from termcolor import colored
import pickle
import torch
import cv2
from unsupervised_rbt.models import ResNetSiameseNetwork
from pyquaternion import Quaternion
from tools.utils import *

import rospy
import yumipy

from phoxipy import PhoXiSensor, ColorizedPhoXiSensor
from ambidex.databases.postgres import YamlLoader, PostgresSchema
from ambidex.class_registry import postgres_base_cls_map, full_cls_list
from ambidex.envs.actions import Grasp3D

from cavity_process import Cavity

class Task(BaseTask):
    def __init__(self, sensor=None):
        super(Task, self).__init__(sensor)
        self.init_camera(sensor)
        self.init_robot()
        self.init_workspace()
        self.init_attributes()

    def init_attributes(self):
        self.base_dir = "physical"

    def good_rotation_state(self,upwards=True):
        self.reset()
        desired_trans = np.array([0.44,-0.12,0.22]) if upwards else np.array([0.44,-0.2,0.35])
        self.robot.right.goto_pose_delta(desired_trans-self.robot.right.get_pose().translation)

    def one_eighty_old(self, x=True):
        self.good_rotation_state()
        num_increment = 10
        angle_increment = 180.0 / num_increment
        rot = [0,angle_increment,0]
        rot2 = [0,-angle_increment,0]
        if x:
            rot = [-angle_increment,0,0]
            rot2 = [angle_increment,0,0]
        for i in range(num_increment):
            self.robot.right.goto_pose_delta([0,0,0],rot)
            self.robot.right.goto_pose_delta([0,0,0.2 / num_increment])
        for i in range(num_increment):
            self.robot.right.goto_pose_delta([0,0,0],rot2)
            self.robot.right.goto_pose_delta([0,0,-0.2 / num_increment])

    def one_eighty_upwards(self, x=True):
        self.good_rotation_state(upwards=True)
        num_increment = 10
        angle_increment = np.pi / num_increment
        rot = [0,-angle_increment,0]
        rot2 = [0,angle_increment,0]
        if x:
            rot = [angle_increment,0,0]
            rot2 = [-angle_increment,0,0]
        for i in range(num_increment):
            rot_transform = Delta_to_Transform([0,0,0.2 / num_increment],rot, self.robot.right.get_pose())
            self.move(rot_transform)
    
    def one_eighty_downwards(self, x=True, num_increment=10):
        trans_delta = 0.22 - self.robot.right.get_pose().translation[2]
        angle_increment = np.pi / num_increment
        rot = [0,-angle_increment,0]
        rot2 = [0,angle_increment,0]
        if x:
            rot = [angle_increment,0,0]
            rot2 = [-angle_increment,0,0]
        for i in range(num_increment):
            rot_transform = Delta_to_Transform([0,0,trans_delta / num_increment],rot2, self.robot.right.get_pose())
            self.move(rot_transform)

        for i in range(num_increment):
            rot_transform = Delta_to_Transform([0,0,-0.2 / num_increment],rot2, self.robot.right.get_pose())
            self.move(rot_transform)

    def align_centroid(self, centroid):
        above_centroid = centroid.copy()
        above_centroid[2] += 0.07
        # above_centroid = np.array([0.443,-0.065,0.32])

        # rot_transform = self.robot.right.get_pose()
        # rot_transform.translation = above_centroid
        # self.robot.right.goto_pose(rot_transform)

        delta = above_centroid - self.robot.right.get_pose().translation
        self.robot.right.goto_pose_delta(delta)
        self.robot.tools['suction'].open_gripper()

def parse_args():
    """Parse arguments from the command line.
    -config to input your own yaml config file. Default is data_gen_quat.yaml
    """
    parser = argparse.ArgumentParser()
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/data_gen_quat.yaml')
    parser.add_argument('-config', type=str, default=default_config_filename)
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    return args

def Get_Transform(rot_quat, ctr_of_mass):
    start_pose = Rotation.from_quat(convert_quat(task.robot.right.get_pose().quaternion, wxyz=True))
    next_pose = Rotation.from_quat(rot_quat) * start_pose
    rot_transform = RigidTransform(RigidTransform.rotation_from_quaternion(convert_quat(next_pose.as_quat(), wxyz=False)), ctr_of_mass)
    return rot_transform, next_pose

if __name__ == "__main__":
    args = parse_args()
    config = YamlConfig(args.config)
    task = Task()

    cavity = Cavity(task.sensor)
    task.robot.tools['suction'].reset()
    cavity_centroid = cavity.process_clamshell_negative()
    task.align_centroid(cavity_centroid)

    # task.one_eighty_ambidex()