import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation
from scipy.stats.morestats import anderson_ksamp
import torch
import torchvision
from autolab_core import YamlConfig, RigidTransform
from pyquaternion import Quaternion
import cv2
from mpl_toolkits.mplot3d import Axes3D
from perception import CameraIntrinsics, RgbdSensorFactory, Image, DepthImage
import os
import trimesh
from .plot_utils import *
from .rotation_utils import *
from .utils import *
from ambidex.databases.postgres import YamlLoader, PostgresSchema
from ambidex.class_registry import postgres_base_cls_map, full_cls_list
from ambidex.envs.actions import Grasp3D
from urdfpy import URDF
import sys
from visualization import Visualizer3D

def demean_preprocess(img):
    depth_min, depth_max = np.min(img[img!=0]), np.max(img)
    img_range = depth_max - depth_min
    img = img.copy()
    img[img!=0] = ((img[img!=0] - depth_min) / img_range / 2) + 0.5
    return img

def segment_robot(robot, pc=None):
    joint_states = robot.right.get_state().joints
    left_states = robot.left.get_state().joints
    sim_robot = URDF.load('physical/yumi_ik/suction_yumi.urdf')
    joint_cfg = {
        "yumi_joint_1_l": left_states[0] / 180 * np.pi,
        "yumi_joint_2_l": left_states[1] / 180 * np.pi,
        "yumi_joint_3_l": left_states[2] / 180 * np.pi,
        "yumi_joint_4_l": left_states[3] / 180 * np.pi,
        "yumi_joint_5_l": left_states[4] / 180 * np.pi,
        "yumi_joint_6_l": left_states[5] / 180 * np.pi,
        "yumi_joint_7_l": left_states[6] / 180 * np.pi,

        "yumi_joint_1_r": joint_states[0] / 180 * np.pi,
        "yumi_joint_2_r": joint_states[1] / 180 * np.pi,
        "yumi_joint_3_r": joint_states[2] / 180 * np.pi,
        "yumi_joint_4_r": joint_states[3] / 180 * np.pi,
        "yumi_joint_5_r": joint_states[4] / 180 * np.pi,
        "yumi_joint_6_r": joint_states[5] / 180 * np.pi,
        "yumi_joint_7_r": joint_states[6] / 180 * np.pi,
    }
    for link in sim_robot.links:
        print(link.name)
    link_fk = sim_robot.link_fk(cfg=joint_cfg)
    sim_robot.show(cfg=joint_cfg)

    visual_fk = sim_robot.visual_trimesh_fk(cfg=joint_cfg)
    # for mesh in visual_fk.keys():
    #     mesh.show()
    suction = visual_fk.keys()[8] #Trimesh obj of suction gripper
    print(suction)
    # suction_pose = visual_fk[suction]
    # suction.apply_transform(suction_pose)
    Visualizer3D.figure()
    # Visualizer3D.mesh(suction, color = (0,255,0))
    for robot_mesh, pose in visual_fk.items():
        robot_mesh.apply_transform(pose)
        Visualizer3D.mesh(robot_mesh)
    Visualizer3D.points(pc, color = (255,0,0), scale = 0.002) if pc is not None else 0
    Visualizer3D.show()

    import pdb
    pdb.set_trace()

    sys.exit()

def depth_to_world_seg(depth, phoxi_intr, T_camera_world, workspace):
    point_cloud_cam = phoxi_intr.deproject(depth)
    point_cloud_cam.remove_zero_points()
    point_cloud_world = T_camera_world * point_cloud_cam
    seg_point_cloud_world, _ = point_cloud_world.box_mask(workspace)
    return seg_point_cloud_world

def world_to_image(point_cloud_world, intr, T_camera_world):
    point_cloud_camera = T_camera_world.inverse() * point_cloud_world
    depth = intr.project_to_image(point_cloud_camera)
    return depth

def segment_depth(depth, phoxi_intr, T_camera_world, workspace):
    seg_point_cloud_world = depth_to_world_seg(depth, phoxi_intr, T_camera_world, workspace)
    depth_seg = world_to_image(seg_point_cloud_world, phoxi_intr, T_camera_world)
    return depth_seg

def plot_depth(depth):
    true_min, true_max = np.min(depth.data[depth.data != 0]), np.max(depth.data)
    depth_subtract = (true_max-true_min) * 0.2
    plt.imshow(depth.data, cmap='gray', vmin=true_min - depth_subtract)
    plt.axis('off')
    plt.show()

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

def convert_quat(quat, wxyz = True):
    if wxyz:
        quat = np.array([quat[1],quat[2],quat[3],quat[0]])
    else:
        quat = np.array([quat[3],quat[0],quat[1],quat[2]])
    return quat


