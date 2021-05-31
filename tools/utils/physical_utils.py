import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation
from scipy.stats.morestats import anderson_ksamp
import torch
import torchvision
from autolab_core import YamlConfig, RigidTransform, TensorDataset, Box, Logger
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
import rospy
import yumipy
from phoxipy import PhoXiSensor, ColorizedPhoXiSensor
import imutils

def Delta_to_Transform(trans, rot, cur_pose):
    rot_matrix = RigidTransform.rotation_from_axis_angle(rot)
    next_rot = rot_matrix.dot(cur_pose.rotation)
    next_trans = cur_pose.translation + trans
    return RigidTransform(next_rot, next_trans)

class BaseTask(object):
    def __init__(self,obj_id=5, plot_images=True):
        Logger.reconfigure_root()
        self.obj_id = obj_id
        self.plot_images = plot_images
    
    def init_attributes(self):
        pass
        
    def init_camera(self, sensor=None):
        calib_dir = '/nfs/diskstation/calib/phoxi'
        phoxi_config = YamlConfig("physical/cfg/tools/colorized_phoxi.yaml")
        self.sensor_name = 'phoxi'
        self.sensor_config = phoxi_config['sensors'][self.sensor_name]

        self.sensor_type = self.sensor_config['type']
        self.sensor_frame = self.sensor_config['frame']
        if sensor:
            self.sensor = sensor
        else:
            self.sensor = ColorizedPhoXiSensor("1703005", 0, calib_dir='/nfs/diskstation/calib/', inpaint=False) 
            self.sensor.start()
        # logger.info('Ready to capture images from sensor %s' %(self.sensor_name))
        self.camera_intr = CameraIntrinsics.load(os.path.join(calib_dir, 'phoxi.intr'))
        self.T_camera_world = RigidTransform.load(os.path.join(calib_dir, 'phoxi_to_world.tf'))

    def init_robot(self):
        basedir = "/home/shivin/catkin_ws/src/ambidex/tests/cfg/"
        yaml_obj_loader = YamlObjLoader(basedir)
        self.robot = yaml_obj_loader('physical_yumi_no_jaw')
        self.home_pose = self.robot.right.get_pose()

        # print "enabling suction \n"
        # self.robot.tools['suction'].close_gripper()
        # time.sleep(2)
        # print "turnin off suction \n"
        # self.robot.tools['suction'].open_gripper()
        # sys.exit()

    def init_workspace(self):
        x,y,z = self.home_pose.translation
        self.workspace = Box(np.array([x-0.066, y-0.066,z-0.065]),
                            np.array([x+0.066,y+0.066, z+0.075]), frame = "world")
        if self.obj_id == 6:
            self.workspace = Box(np.array([x-0.075, y-0.075,z-0.065]),
                            np.array([x+0.075,y+0.075, z+0.075]), frame = "world")
        if self.obj_id == 7:
            self.workspace = Box(np.array([x-0.075, y-0.075,z-0.065]),
                            np.array([x+0.075,y+0.075, z+0.075]), frame = "world")
        #CASE 2020 was 0.1, 0.075, -0.05 - 0.1

    def capture_image(self, center_i=None, center_j=None, frame=0):
        # logger.info('Capturing depth image' + str(frame))
        img = self.sensor.read()
        rgb_1, depth_1 = img.color, img.depth

        depth_2 = segment_depth(depth_1,self.camera_intr,self.T_camera_world,self.workspace)
        segmask = depth_2.to_binary()
        rgb_2 = rgb_1.mask_binary(segmask)

        rgb_3, depth_3 = self.color_mask(rgb_2,depth_2)

        if self.obj_id == 6:
            depth_4 = crop_image(depth_3,crop_height=280)
        else:
            depth_4 = crop_image(depth_3,bb_center=True, bb_slack=11) #TODO check for center i and j
        # depth_seg4.save('ros_phoxi/depth_seg4_%d.png' %(frame))
        I_sg = depth_4.data
        # mask = I_sg != 0
        # I_sg[mask] = I_sg[mask] - np.min(depth_seg4.data)
        # I_sg[mask] = (I_sg[mask] / np.max(I_sg[mask]) * 0.1) + 0.5
        frame_string = str(frame).zfill(2)
        if self.plot_images:
            Plot_Image(rgb_1.data, self.base_path + "/rgb_1/" + frame_string + ".png")
            Plot_Image(rgb_2.data, self.base_path + "/rgb_2/" + frame_string + ".png")
            Plot_Image(rgb_3.data, self.base_path + "/rgb_3/" + frame_string + ".png")
            Plot_Image(depth_1.data, self.base_path + "/depth_1/" + frame_string + ".png")
            Plot_Image(depth_2.data, self.base_path + "/depth_2/" + frame_string + ".png")
            Plot_Image(depth_3.data, self.base_path + "/depth_3/" + frame_string + ".png")
        Plot_Image(depth_4.data, self.base_path + "/depth_4/" + frame_string + ".png")

        # I_sg = demean_preprocess(I_sg)
        seg_pc = depth_to_world_seg(depth_3,self.camera_intr,self.T_camera_world,self.workspace)
        return I_sg, seg_pc
    
    def color_mask(self, rgb_2, depth_2):
        # color_mask = rgb_2.segment_hsv(np.array([80,60,0]), np.array([120,255,255])) #blue-green pipe connector
        # color_mask = rgb_2.segment_hsv(np.array([90,60,0]), np.array([140,255,255])) #purple clamp
        # color_mask = rgb_2.segment_hsv(np.array([0,0,1]), np.array([140,255,70])) #black tube
        # color_mask = rgb_2.segment_hsv(np.array([0,100,1]), np.array([25,255,255])) #brown rock climbing hold
        # color_mask = rgb_2.segment_hsv(np.array([80,60,0]), np.array([120,255,255])) #blue barclamp

        color_mask = rgb_2.segment_hsv(np.array([0,0,115]), np.array([180,110,255])) #white handrail bracket
        if self.obj_id == 6:
            color_mask = rgb_2.segment_hsv(np.array([0,0,75]), np.array([180,255,255])) #silver ornamental handrail bracket

        rgb_3 = rgb_2.mask_binary(color_mask, invert=False)
        depth_3 = depth_2.mask_binary(color_mask, invert=False)

        # rgb_3, depth_3 = rgb_2, depth_2
        return rgb_3, depth_3

    def move(self, rot_transform):
        """params: all in frame of object --> world
        Moves gripper"""
        self.robot.tools['suction'].move_to(rot_transform)

    def reset(self):
        self.robot.tools['suction'].reset()

    def reset_home_pose_workspace(self):
        self.home_pose = self.robot.right.get_pose()
        self.init_workspace()
        return self.home_pose

    def finish(self):
        # self.robot.right.open_gripper()
        self.robot.stop() # Stop the robot
        # self.sensor.stop() # Stop the phoxi

# def best_rotation(image1, image2):
#     0

def Plot_Physical_Loss(losses, base_path):
    plt.plot(losses)
    plt.title("Angle Difference Between Iteration Orientation and Goal Orientation")
    plt.ylabel("Angle Difference (Degrees)")
    plt.xlabel("Iteration Number")
    plt.savefig(base_path + "/loss.png")
    plt.close()
    np.save(base_path + "/losses.npy",losses)

def filter_crop_image(depth, thresh = 1000.0, crop_height=256):
    depth = remove_junk(depth, thresh)
    return crop_image(depth, crop_height)

def remove_junk(depth, thresh=1000.0):
    binary1 = depth.to_binary()
    binary2 = binary1.prune_contours(area_thresh=thresh, dist_thresh=0)
    return depth.mask_binary(binary2)

def crop_image(depth, crop_height=None, bb_center=None, bb_slack = None):
    ind = np.where(depth.data != 0)
    if bb_center:
        center_i, center_j = (np.max(ind[0]) + np.min(ind[0])) / 2, (np.max(ind[1]) + np.min(ind[1])) / 2 
    else:
        center_i, center_j = np.mean(ind[0]), np.mean(ind[1]) 

    if crop_height:
        height, width = crop_height,crop_height
    elif bb_slack:
        height, width = np.max(ind[0])- np.min(ind[0]), np.max(ind[1])- np.min(ind[1])
        height, width = (height*bb_slack)/10, (width*bb_slack)/10
        height, width = max((height,width)), max((height,width))
    else:
        height, width = 256, 256

    depth_int = depth.crop(height, width, center_i, center_j)
    depth = depth_int.resize((128,128), 'nearest')
    return depth

def Make_Directories(base_path, iteration):
    cur_dir = base_path + "/iteration" + str(iteration)
    for folder in ["rgb_1","rgb_2","rgb_3","depth_1","depth_2","depth_3","depth_4", "poses"]:
        next_dir = cur_dir + "/" + folder + "/"
        if not os.path.exists(os.path.dirname(next_dir)):
            print("Making path", next_dir)
            os.makedirs(os.path.dirname(next_dir))
    return cur_dir

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


