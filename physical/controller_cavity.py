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

        x,y,z = self.home_pose.translation
        self.workspace = Box(np.array([x-0.1, y-0.075,z-0.05]),
                            np.array([x+0.1,y+0.075, z+0.10]), frame = "world")

        Logger.reconfigure_root()

        self.save_dir = "ros_phoxi"
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        # print "enabling suction \n"
        # self.robot.tools['suction'].close_gripper()
        # time.sleep(2)
        # print "turnin off suction \n"
        # self.robot.tools['suction'].open_gripper()
        # sys.exit()

    def capture_image(self, center_i=None, center_j=None, frame=0):
        # logger.info('Capturing depth image' + str(frame))
        img = self.sensor.read()
        rgb_1, depth_1 = img.color, img.depth

        # deproject into 3D world coordinates
        point_cloud_cam = self.camera_intr.deproject(depth_1)
        point_cloud_cam.remove_zero_points()
        point_cloud_world = self.T_camera_world * point_cloud_cam
        seg_point_cloud_world, _ = point_cloud_world.box_mask(self.workspace)
        
        # compute the segmask for points above the box
        seg_point_cloud_cam = self.T_camera_world.inverse() * seg_point_cloud_world
        depth_2 = self.camera_intr.project_to_image(seg_point_cloud_cam)
        segmask = depth_2.to_binary()
        rgb_2 = rgb_1.mask_binary(segmask)

        rgb_3, depth_3 = self.color_mask(rgb_2,depth_2)
        ind = np.where(depth_3.data != 0)
        if not center_i or not center_j:
            center_i, center_j = (np.max(ind[0]) + np.min(ind[0])) / 2, (np.max(ind[1]) + np.min(ind[1])) / 2 
        height, width = np.max(ind[0])- np.min(ind[0]), np.max(ind[1])- np.min(ind[1])
        height, width = (height*11)/10, (width*11)/10
        height, width = max((height,width)), max((height,width))

        print(height, width, center_i, center_j)
        
        depth_int = depth_3.crop(height, width, center_i, center_j)
        depth_4 = depth_int.resize((128,128), 'nearest')
        # depth_seg4.save('ros_phoxi/depth_seg4_%d.png' %(frame))
        I_sg = depth_4.data
        # mask = I_sg != 0
        # I_sg[mask] = I_sg[mask] - np.min(depth_seg4.data)
        # I_sg[mask] = (I_sg[mask] / np.max(I_sg[mask]) * 0.1) + 0.5
        frame_string = str(frame).zfill(2)
        Plot_Image(rgb_1.data, base_path + "/rgb_1/" + frame_string + ".png")
        Plot_Image(rgb_2.data, base_path + "/rgb_2/" + frame_string + ".png")
        Plot_Image(rgb_3.data, base_path + "/rgb_3/" + frame_string + ".png")
        Plot_Image(depth_1.data, base_path + "/depth_1/" + frame_string + ".png")
        Plot_Image(depth_2.data, base_path + "/depth_2/" + frame_string + ".png")
        Plot_Image(depth_3.data, base_path + "/depth_3/" + frame_string + ".png")
        Plot_Image(depth_4.data, base_path + "/depth_4/" + frame_string + ".png")

        # I_sg = demean_preprocess(I_sg)

        return I_sg, center_i, center_j
    
    def color_mask(self, rgb_2, depth_2):
        # color_mask = rgb_2.segment_hsv(np.array([80,60,0]), np.array([120,255,255])) #blue-green pipe connector
        # color_mask = rgb_2.segment_hsv(np.array([90,60,0]), np.array([140,255,255])) #purple clamp
        # color_mask = rgb_2.segment_hsv(np.array([0,0,1]), np.array([140,255,70])) #black tube
        # color_mask = rgb_2.segment_hsv(np.array([0,100,1]), np.array([25,255,255])) #brown rock climbing hold
        # color_mask = rgb_2.segment_hsv(np.array([80,60,0]), np.array([120,255,255])) #blue barclamp
        color_mask = rgb_2.segment_hsv(np.array([0,0,100]), np.array([180,120,255])) #gold handrail bracket

        rgb_3 = rgb_2.mask_binary(color_mask, invert=False)
        depth_3 = depth_2.mask_binary(color_mask, invert=False)

        # rgb_3, depth_3 = rgb_2, depth_2
        return rgb_3, depth_3

    def move(self, rot_transform):
        """params: all in frame of object --> world
        Moves gripper"""
        self.robot.right.goto_pose(rot_transform)

    def finish(self):
        # self.robot.right.open_gripper()
        self.robot.stop() # Stop the robot
        # self.sensor.stop() # Stop the phoxi

def Plot_Image(image, filename):
    """x
    """
    # cv2.imwrite(filename, image)
    img_range = np.max(image) - np.min(image[image != 0]) + 0.0001
    plt.imshow(image, cmap='gray', vmin = np.min(image[image != 0]) - img_range * 0.1, vmax = np.max(image))
    plt.axis('off')
    plt.savefig(filename)
    # plt.show()
    plt.close()

def Save_Poses(pose_quat, index):
    """x
    """
    np.savetxt(base_path + "/poses/quat_" + index + ".txt", pose_quat)

def Make_Directories(base_path, iteration):
    cur_dir = base_path + "/iteration" + str(iteration)
    for folder in ["rgb_1","rgb_2","rgb_3","depth_1","depth_2","depth_3","depth_4", "poses"]:
        next_dir = cur_dir + "/" + folder + "/"
        if not os.path.exists(os.path.dirname(next_dir)):
            print("Making path", next_dir)
            os.makedirs(os.path.dirname(next_dir))
    return cur_dir

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
    # logger = Logger.get_logger('physical_controller.py')

    obj_id = 5
    max_iterations = 20
    iteration = 0
    base_path = "physical/objects/obj" + str(obj_id)
    base_path = Make_Directories(base_path, iteration)
    # INITIALIZE TO (440, -23, 281) TRANSLATION

    print(colored('---------- Starting for Object ' + str(obj_id) + ' Iteration ' + str(iteration) +'----------', 'red'))
    cavity = Cavity()
    # Render image 2, which will be the goal image of the object in a stable pose
    # I_g, center_i, center_j = task.capture_image(frame= "goal")
    # I_g = cavity.process_clamshell_positive()
    I_g = cavity.get_goal_img()
    Plot_Image(I_g, base_path + "/depth_4/goal.png")

    # raw_input("Grasp object in correct pose ")
    task = Task(sensor=cavity.sensor)
    # task.robot.tools['suction'].close_gripper()
    # segment_robot(task.robot)

    ctr_of_mass = task.robot.right.get_pose().translation
    # assert ctr_of_mass[1] > -0.130
    start_pose = Rotation.from_quat(convert_quat(task.robot.right.get_pose().quaternion, wxyz=True))
    # next_pose = Rotation.from_rotvec(np.array([np.pi,0,0])) * start_pose #TODO this is weird
    next_pose = Rotation.from_rotvec(np.array([0,np.pi,0])) * start_pose #TODO this is weird
    translated_c_o_m = ctr_of_mass.copy()
    # translated_c_o_m[2] = 0.32
    rot_transform = RigidTransform(RigidTransform.rotation_from_quaternion(convert_quat(next_pose.as_quat(), wxyz=False)), translated_c_o_m)
    task.robot.right.goto_pose(rot_transform)
    sys.exit()
    poses = {}

    goal_pose = Rotation.from_quat(convert_quat(task.robot.right.get_pose().quaternion, wxyz=True))
    # ctr_of_mass = task.robot.right.get_pose().translation
    # Save_Poses(goal_pose.as_quat(), "goal")
    poses['goal'] = goal_pose.as_quat()


    # Render image 1, which will be 30 degrees away from the goal
    rot_quat = Generate_Quaternion(19.9/180*np.pi,20.0/180*np.pi)
    rot_transform, start_pose = Get_Transform(rot_quat, ctr_of_mass)

    print("Robot starts at quat: ", Quaternion_String(goal_pose.as_quat()))
    print("Rotating by Quat:", rot_quat)
    print("Rotating by Euler Angles:", Rotation.from_quat(rot_quat).as_euler('xyz', degrees=True))
    print("Robot should be at quat: ", Quaternion_String(start_pose.as_quat()))

    print(rot_transform.quaternion, rot_transform.translation)

    task.robot.right.goto_pose(rot_transform)
    start_pose = Rotation.from_quat(convert_quat(task.robot.right.get_pose().quaternion, wxyz=True))
    # Save_Poses(start_pose.as_quat(), "0")
    poses[0] = start_pose.as_quat()

    I_s, center_i, center_j = task.capture_image(frame=0)
    # I_s, _, _ = task.capture_image(center_i, center_j,frame=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetSiameseNetwork().to(device)
    model.load_state_dict(torch.load("models/new_datagen_16.pt"))
    model.eval()

    im1_batch = torch.Tensor(torch.from_numpy(I_s).float()).to(device).unsqueeze(0).unsqueeze(0)
    im2_batch = torch.Tensor(torch.from_numpy(I_g).float()).to(device).unsqueeze(0).unsqueeze(0)
    # print(im1_batch.size())
    # print(im2_batch.size())
    total_iters = 1
    cur_pose = start_pose
    for i in range(1,max_iterations):
        pred_quat = model(im1_batch,im2_batch).detach().cpu().numpy()[0]
        angle = np.arccos(pred_quat[3])*2 * 180 / np.pi
        print("Angle to rotate:", angle)
        # if pred_quat[3] >= 0.9998476952: # 2 degrees
        if pred_quat[3] >= 0.99996192306: # 1 degree
        # if pred_quat[3] >= 0.99999048128: # 0.5 degrees
        # if pred_quat[3] >= 1: # 0 degrees
            print("Stopping Criteria:", np.arccos(pred_quat[3]) * 180 /np.pi*2, pred_quat)
            break
        if angle < 1.25:
            slerp = 1 if angle < 1.0 else 0.6*angle
        else:
            slerp = 0.6
        pred_quat = Quaternion(convert_quat(pred_quat, wxyz=False))
        rot_quat = Quaternion.slerp(Quaternion(), pred_quat, slerp).elements # 1 is pred quat, 0 is no rot
        rot_quat = normalize(convert_quat(rot_quat, wxyz=True))
        rot_transform, cur_pose = Get_Transform(rot_quat, ctr_of_mass)
        task.robot.right.goto_pose(rot_transform)
        cur_pose = Rotation.from_quat(convert_quat(task.robot.right.get_pose().quaternion, wxyz=True))
        # Save_Poses(cur_pose.as_quat(), str(i))
        poses[i] = cur_pose.as_quat()

        cur_image, _, _ = task.capture_image(center_i, center_j, i)
        # cur_image = (cur_image*65535).astype(int)
        im1_batch = torch.Tensor(torch.from_numpy(cur_image).float()).to(device).unsqueeze(0).unsqueeze(0)
        total_iters = i
        losses = []
        quat_goal = poses['goal']
        # print(quat_goal)
        for j in range(total_iters):
            q = poses[j]
            # print(q)
            # print(np.dot(q,quat_goal))
            angle_error = np.arccos(np.abs(np.dot(quat_goal, q)))*180/np.pi*2
            if j == total_iters - 1:
                print("Current Error: ", angle_error)
            losses.append(angle_error)
        # print(np.round(losses,2)[::5])
        plt.plot(losses)
        plt.title("Angle Difference Between Iteration Orientation and Goal Orientation")
        plt.ylabel("Angle Difference (Degrees)")
        plt.xlabel("Iteration Number")
        plt.savefig(base_path + "/loss.png")
        plt.close()
        np.save(base_path + "/losses.npy",losses)
    task.finish()