'''
This script generates data for the self-supervised rotation prediction task
'''

from autolab_core import YamlConfig, RigidTransform, TensorDataset, Box, Logger
from scipy.spatial.transform import Rotation
import os

import numpy as np
import itertools
import sys
import argparse

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt

import random
from termcolor import colored
import pickle
import torch
import cv2
from unsupervised_rbt.models import ResNetSiameseNetwork, InceptionSiameseNetwork
from pyquaternion import Quaternion
from tools.utils import *

import rospy
import yumipy

def convert_quat(quat, wxyz = True):
    if wxyz:
        quat = np.array([quat[1],quat[2],quat[3],quat[0]])
    else:
        quat = np.array([quat[3],quat[0],quat[1],quat[2]])
    return quat

class Task:
    def __init__(self, calib_dir):
        self.robot = yumipy.YuMiRobot(include_right=False)
        self.home_pose = self.robot.left.get_pose()
        self.camera_intr = CameraIntrinsics.load(os.path.join(calib_dir, 'phoxi.intr'))
        self.T_camera_world = RigidTransform.load(os.path.join(calib_dir, 'phoxi_to_world.tf'))
        # self.robot.left.close_gripper()

        # INITIALIZE PHOXI CAMERA
        phoxi_config = YamlConfig("physical/cfg/tools/colorized_phoxi.yaml")

        x,y,z = self.home_pose.translation
        self.workspace = Box(np.array([x-0.1, y-0.075,z-0.05]),
                            np.array([x+0.1,y+0.075, z+0.15]), frame = "world")

        rospy.init_node('capture_test_images') #NOTE: this is required by the camera sensor classes
        Logger.reconfigure_root()

        self.sensor_name = 'phoxi'
        self.sensor_config = phoxi_config['sensors'][self.sensor_name]
        # self.demo_horizon_length = phoxi_config['sensors'][self.sensor_name]['num_images']

        # logger.info('Ready to capture images from sensor %s' %(self.sensor_name))
        self.save_dir = "ros_phoxi"
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)        
        
        # read params
        self.sensor_type = self.sensor_config['type']
        self.sensor_frame = self.sensor_config['frame']
        self.sensor = RgbdSensorFactory.sensor(self.sensor_type, self.sensor_config)
        self.sensor.start()

    def capture_image(self, center_i=None, center_j=None, frame=0):
        # logger.info('Capturing depth image' + str(frame))
        rgb_1, depth_1, _ = self.sensor.frames()

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

        # color_mask = rgb_2.segment_hsv(np.array([80,60,0]), np.array([120,255,255]))
        # color_mask = rgb_2.segment_hsv(np.array([90,60,0]), np.array([140,255,255]))
        color_mask = rgb_2.segment_hsv(np.array([0,0,1]), np.array([140,255,70]))
        # color_mask = rgb_2.segment_hsv(np.array([0,100,1]), np.array([25,255,255]))
        # color_mask = rgb_2.segment_hsv(np.array([80,60,0]), np.array([120,255,255]))

        rgb_3 = rgb_2.mask_binary(color_mask, invert=False)
        depth_3 = depth_2.mask_binary(color_mask, invert=False)
        ind = np.where(depth_3.data != 0)
        if not center_i or not center_j:
            center_i, center_j = np.mean(ind[0]), np.mean(ind[1])
        height, width = np.max(ind[0])- np.min(ind[0]), np.max(ind[1])- np.min(ind[1])
        # print(height, width, center_i, center_j)
        depth_int = depth_3.crop(256, 256, center_i, center_j)
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

        return I_sg, center_i, center_j
    
    def move(self, rot_transform):
        """params: all in frame of object --> world
        Moves gripper"""
        self.robot.left.goto_pose(rot_transform)

    def finish(self):
        # self.robot.left.open_gripper()
        self.robot.stop() # Stop the robot
        self.sensor.stop() # Stop the phoxi

def Plot_Image(image, filename):
    """x
    """
    # cv2.imwrite(filename, image)
    fig1 = plt.imshow(image, cmap='gray', vmin = np.min(image[image != 0])*0.9, vmax = np.max(image))
    fig1.axes.get_xaxis().set_visible(False)
    fig1.axes.get_yaxis().set_visible(False)
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
    start_pose = Rotation.from_quat(convert_quat(task.robot.left.get_pose().quaternion, wxyz=True))
    next_pose = Rotation.from_quat(rot_quat) * start_pose
    rot_transform = RigidTransform(RigidTransform.rotation_from_quaternion(convert_quat(next_pose.as_quat(), wxyz=False)), ctr_of_mass)
    return rot_transform, next_pose

if __name__ == "__main__":
    args = parse_args()
    config = YamlConfig(args.config)
    # logger = Logger.get_logger('physical_controller.py')

    obj_id = 2
    max_iterations = 50
    iteration = 11
    base_path = "physical/objects/obj" + str(obj_id)
    base_path = Make_Directories(base_path, iteration)
    # INITIALIZE TO (440, -23, 281) TRANSLATION

    print(colored('---------- Starting for Object ' + str(obj_id) + ' Iteration ' + str(iteration) +'----------', 'red'))
    task = Task('/nfs/diskstation/calib/phoxi')

    goal_pose = Rotation.from_quat(convert_quat(task.robot.left.get_pose().quaternion, wxyz=True))
    ctr_of_mass = task.robot.left.get_pose().translation
    Save_Poses(goal_pose.as_quat(), "goal")

    # Render image 2, which will be the goal image of the object in a stable pose
    I_g, center_i, center_j = task.capture_image(frame= "goal")

    # Render image 1, which will be 30 degrees away from the goal
    rot_quat = Generate_Quaternion(29.9/180 *np.pi,np.pi/6)
    rot_transform, start_pose = Get_Transform(rot_quat, ctr_of_mass)

    print("Robot starts at quat: ", Quaternion_String(goal_pose.as_quat()))
    print("Rotating by Quat:", rot_quat)
    print("Rotating by Euler Angles:", Rotation.from_quat(rot_quat).as_euler('xyz', degrees=True))
    print("Robot should be at quat: ", Quaternion_String(start_pose.as_quat()))

    print(rot_transform.quaternion, rot_transform.translation)

    task.robot.left.goto_pose(rot_transform)
    start_pose = Rotation.from_quat(convert_quat(task.robot.left.get_pose().quaternion, wxyz=True))
    Save_Poses(start_pose.as_quat(), "0")

    I_s, _, _ = task.capture_image(center_i, center_j, 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetSiameseNetwork(4, n_blocks=1, embed_dim=1024, dropout=4).to(device)
    model.load_state_dict(torch.load("models/872obj/cos_sm_blk1_emb1024_reg9_drop4.pt"))
    model.eval()

    im1_batch = torch.Tensor(torch.from_numpy(I_s).float()).to(device).unsqueeze(0).unsqueeze(0)
    im2_batch = torch.Tensor(torch.from_numpy(I_g).float()).to(device).unsqueeze(0).unsqueeze(0)
    # print(im1_batch.size())
    total_iters = 1
    cur_pose = start_pose
    for i in range(1,max_iterations):
        pred_quat = model(im1_batch,im2_batch).detach().cpu().numpy()[0]
        angle = np.arccos(pred_quat[3])*2 * 180 / np.pi
        print("Angle to rotate:", angle)
        # if pred_quat[3] >= 0.9998476952: # 2 degrees
        # if pred_quat[3] >= 0.99996192306: # 1 degree
        if pred_quat[3] >= 0.99999048128: # 0.5 degrees
        # if pred_quat[3] >= 1: # 0 degrees
            print("Stopping Criteria:", np.arccos(pred_quat[3]) * 180 /np.pi*2, pred_quat)
            break
        if angle < 2.5:
            slerp = 1 if angle < 0.5 else 0.5/angle
        else:
            slerp = 0.2
        pred_quat = Quaternion(convert_quat(pred_quat, wxyz=False))
        rot_quat = Quaternion.slerp(Quaternion(), pred_quat, slerp).elements # 1 is pred quat, 0 is no rot
        rot_quat = normalize(convert_quat(rot_quat, wxyz=True))
        rot_transform, cur_pose = Get_Transform(rot_quat, ctr_of_mass)
        task.robot.left.goto_pose(rot_transform)
        cur_pose = Rotation.from_quat(convert_quat(task.robot.left.get_pose().quaternion, wxyz=True))
        Save_Poses(cur_pose.as_quat(), str(i))

        cur_image, _, _ = task.capture_image(center_i, center_j, i)
        # cur_image = (cur_image*65535).astype(int)
        im1_batch = torch.Tensor(torch.from_numpy(cur_image).float()).to(device).unsqueeze(0).unsqueeze(0)
        total_iters = i
        losses = []
        quat_goal = np.loadtxt(base_path + "/poses/quat_goal.txt")
        # print(quat_goal)
        for j in range(total_iters):
            q = np.loadtxt(base_path + "/poses/quat_" + str(j) + ".txt")
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