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

from pose_estimator import PoseEstimator, demean

class Task(BaseTask):
    def __init__(self, obj_id, plot_images, base_path, sensor=None):
        super(Task, self).__init__(obj_id, plot_images)
        self.init_camera(sensor)
        self.init_robot()
        self.init_workspace()
        self.init_attributes(base_path)

    def init_attributes(self, base_path):
        self.base_path = base_path

    def good_rotation_state(self,upwards=True):
        if upwards:
            desired_trans = np.array([0.40,-0.12,0.22])
        else:
            desired_trans = np.array([0.40,-0.12,0.35])
            desired_trans[0:2] = self.robot.right.get_pose().translation[:2] #TODO better way of doing this
        next_pose = self.robot.right.get_pose().copy()
            # next_pose.translation = desired_trans
            # self.move(next_pose)
        next_pose.translation = desired_trans
        self.move(next_pose)

        # self.robot.right.goto_pose_delta(desired_trans-self.robot.right.get_pose().translation)
        # cur_pose = self.reset_home_pose_workspace()

        return next_pose

    def one_eighty_upwards(self, next_pose):
        # trans_delta = 0.35 - self.robot.right.get_pose().translation[2]
        # self.move(Delta_to_Transform([0,0,trans_delta],[np.pi,0,0],self.robot.right.get_pose()))

        trans_delta = 0.35 - next_pose.translation[2]
        self.move(Delta_to_Transform([0,0,trans_delta],[np.pi,0,0],next_pose))
        
        # self.robot.right.goto_pose_delta([0,0,-0.07])
        cur_pose = self.reset_home_pose_workspace()

    def one_eighty_downwards(self,num_incr=3):
        trans_delta = 0.22 - self.robot.right.get_pose().translation[2]
        for i in range(num_incr):
            self.move(Delta_to_Transform([0,0,trans_delta/num_incr],[-np.pi/num_incr,0,0],self.robot.right.get_pose()))
        # cur_pose = task.reset_home_pose_workspace()

    def align_centroid(self, centroid, pc_center):
        above_centroid = centroid.copy()
        above_centroid[2] += 0.07 if self.obj_id != 7 else 0.09
        # above_centroid = np.array([0.443,-0.065,0.32])

        # rot_transform = self.robot.right.get_pose()
        # rot_transform.translation = above_centroid
        # self.robot.right.goto_pose(rot_transform)
        self.robot.right.goto_pose_delta([0,0,0.35 - self.robot.right.get_pose().translation[2]])

        delta = above_centroid - pc_center
        delta[2] = above_centroid[2] - self.robot.right.get_pose().translation[2]
        self.robot.right.goto_pose_delta(delta)
        self.robot.tools['suction'].open_gripper()

    def insert_positive(self, cavity, pc_center):
        cavity_pc = cavity.process_clamshell_negative()
        cavity_centroid = cavity_pc.mean().data
        next_pose = self.good_rotation_state(upwards=False)
        self.one_eighty_downwards()
        self.align_centroid(cavity_centroid, pc_center)

    def insert_negative(self, cavity_centroid, pc_center):
        self.good_rotation_state(upwards=False)
        self.one_eighty_downwards()
        self.align_centroid(cavity_centroid, pc_center)

    def get_rotation(self, pc1, pc2):
        estimator = PoseEstimator(self.camera_intr, self.T_camera_world,self.workspace)
        pc1,_ = pc1.subsample(pc1.num_points / 500)
        pc2,_ = pc2.subsample(pc2.num_points / 500)
        pc1 = torch.Tensor(demean(pc1.data)).float().to("cuda")
        pc2 = torch.Tensor(demean(pc2.data)).float().to("cuda")
        return estimator._is_match(pc1, pc2)

    def network_orient(self, I_g, poses, max_iterations):
        cur_pose = self.reset_home_pose_workspace()
        ctr_of_mass = cur_pose.translation

        I_s, cur_pc = self.capture_image(frame=0)
        pc_center = cur_pc.mean().data

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetSiameseNetwork().to(device)
        model.load_state_dict(torch.load("models/872objv4_17.pt"))
        # model.load_state_dict(torch.load("models/872objv4_shuffled_15.pt"))
        model.eval()

        im1_batch = torch.Tensor(torch.from_numpy(I_s).float()).to(device).unsqueeze(0).unsqueeze(0)
        im2_batch = torch.Tensor(torch.from_numpy(I_g).float()).to(device).unsqueeze(0).unsqueeze(0)
        # print(im1_batch.size())
        # print(im2_batch.size())

        angle_error = np.arccos(np.abs(np.dot(poses[0], poses[1])))*180/np.pi*2
        print("Iter 0 Current Error: ", angle_error)
        losses = [angle_error]
        for i in range(1,max_iterations):
            pred_quat = model(im1_batch,im2_batch).detach().cpu().numpy()[0]
            angle = np.arccos(pred_quat[3])*2 * 180 / np.pi
            print("Angle to rotate:", angle)

            slerp = Get_Slerp(0.8, 2.0, angle)
            rot_transform, cur_pose = Slerp_to_Transform(slerp, pred_quat, ctr_of_mass)
            self.move(rot_transform)
            cur_pose = Rotation.from_quat(convert_quat(self.robot.right.get_pose().quaternion, wxyz=True))
            poses[i+1] = cur_pose.as_quat()

            cur_image, cur_pc = self.capture_image(frame=i)
            pc_center = cur_pc.mean().data
            im1_batch = torch.Tensor(torch.from_numpy(cur_image).float()).to(device).unsqueeze(0).unsqueeze(0)
            quat_goal = poses[0]
            for j in range(i+1):
                q = poses[j+1]
                angle_error = np.arccos(np.abs(np.dot(quat_goal, q)))*180/np.pi*2
                if j == i:
                    print("Iter",i,"Current Error: ", angle_error)
                    losses.append(angle_error)
            if pred_quat[3] >= 0.99904822158: # 5 degrees
            # if pred_quat[3] >= 0.99939082701: # 4 degrees
            # if pred_quat[3] >= 0.9998476952: # 2 degrees
                print("Stopping Criteria:", np.arccos(pred_quat[3]) * 180 /np.pi*2, pred_quat)
                break
        return losses, pc_center

    def baseline_orient(self, cavity_pc):
        cur_pose = self.reset_home_pose_workspace()
        ctr_of_mass = cur_pose.translation
        I_s, cur_pc = self.capture_image(frame=0)
        rot = self.get_rotation(cur_pc, cavity_pc)
        print("Predicted rotation of", rot * 180 / np.pi, "around z-axis")
        rot_transform = Delta_to_Transform(np.array([0,0,0]),np.array([0,0,1])*rot,self.robot.right.get_pose())
        self.move(rot_transform)
        cur_image, cur_pc = self.capture_image(frame=1)
        pc_center = cur_pc.mean().data
        return pc_center

    def apply_random_rot(self,rot_angle=30.0):
        cur_pose = self.reset_home_pose_workspace()
        ctr_of_mass = cur_pose.translation
        rot_quat = Generate_Quaternion((rot_angle - 0.1)/180*np.pi,rot_angle/180*np.pi)
        next_pose, start_pose = Get_Transform(rot_quat, ctr_of_mass)
        # self.move(next_pose)
        return next_pose

    def align_for_orienting(self,cavity_centroid):
        next_pose = self.robot.right.get_pose().copy()
        next_pose.translation[2] = cavity_centroid[2] #- 0.005
        self.move(next_pose)


def Save_Poses(pose_quat, index):
    """x
    """
    np.savetxt(base_path + "/poses/quat_" + index + ".txt", pose_quat)

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

def Get_Slerp(default_slerp, min_angle, rot_angle):
    if rot_angle < min_angle / default_slerp:
        slerp = 1 if rot_angle < min_angle else default_slerp*rot_angle / min_angle
    else:
        slerp = default_slerp
    return slerp

def Slerp_to_Transform(slerp, pred_quat, ctr_of_mass):
    pred_quat = Quaternion(convert_quat(pred_quat, wxyz=False))
    rot_quat = Quaternion.slerp(Quaternion(), pred_quat, slerp).elements # 1 is pred quat, 0 is no rot
    rot_quat = normalize(convert_quat(rot_quat, wxyz=True))
    rot_transform, cur_pose = Get_Transform(rot_quat, ctr_of_mass)
    return rot_transform, cur_pose
        
if __name__ == "__main__":
    args = parse_args()
    config = YamlConfig(args.config)

    obj_id = 7
    max_iterations = 8
    iteration = 9
    positive = 0
    rot_angle = 30.0
    plot_images = 0

    base_path = "physical/objects/obj" + str(obj_id)
    base_path = Make_Directories(base_path, iteration)
    print(colored('---------- Starting for Object ' + str(obj_id) + ' Iteration ' + str(iteration) +'----------', 'red'))
    cavity = Cavity(plot_images=plot_images)

    # Render image 2, which will be the goal image of the object
    # I_g, center_i, center_j = task.capture_image(frame= "goal")
    # I_g = cavity.process_clamshell_positive()

    task = Task(obj_id,plot_images, base_path, sensor=cavity.sensor)
    
    # segment_robot(task.robot)

    task.reset()
    task.robot.tools['suction'].close_gripper()
    # sys.exit()
    raw_input("Grasp object ")

    if positive:
        I_g, cavity_pc = cavity.get_goal_img(obj_id)
        cavity_centroid = cavity_pc.mean().data
    else:
        cavity_pc = cavity.process_clamshell_negative(elevated=True)
        I_g, cavity_centroid = cavity.rotate_negative(cavity_pc)
    print("Cavity was centered at", cavity_centroid)

    Plot_Image(I_g, base_path + "/depth_4/goal.png")
    
    poses = np.zeros((max_iterations+1,4))
    initial_pose = Rotation.from_quat(convert_quat(task.robot.right.get_pose().quaternion, wxyz=True))
    goal_pose = Rotation.from_rotvec(np.array([np.pi,0,0])) * initial_pose
    poses[0] = goal_pose.as_quat()

    # Rotate object 30 degrees away from the goal
    next_pose = task.good_rotation_state(upwards=True)
    next_pose = task.apply_random_rot(rot_angle)
    # task.move(next_pose)
    # time.sleep(0.5)

    task.one_eighty_upwards(next_pose)
    task.align_for_orienting(cavity_centroid)

    # task.good_rotation_state(upwards=False)
    # task.one_eighty_downwards()

    # print("Robot starts at quat: ", Quaternion_String(goal_pose.as_quat()))
    # print("Rotating by Quat:", rot_quat)
    # print("Rotating by Euler Angles:", Rotation.from_quat(rot_quat).as_euler('xyz', degrees=True))
    # print("Robot should be at quat: ", Quaternion_String(start_pose.as_quat()))

    # start_pose = Rotation.from_quat(convert_quat(task.robot.right.get_pose().quaternion, wxyz=True))
    # poses[1] = start_pose.as_quat()

    pc_center = task.baseline_orient(cavity_pc)

    # losses,pc_center = task.network_orient(I_g,poses,max_iterations)
    # Plot_Physical_Loss(losses, base_path)
    # print(np.round(losses,2)[::5])

    np.savetxt(base_path + "/poses.txt", poses)
    task.insert_positive(cavity, pc_center) if positive else task.insert_negative(cavity_centroid, pc_center)
    task.finish()