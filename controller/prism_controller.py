from autolab_core import YamlConfig, RigidTransform, TensorDataset
from scipy.spatial.transform import Rotation
import os

# Use this if you are SSH
os.environ["PYOPENGL_PLATFORM"] = 'egl'
# os.environ["PYOPENGL_PLATFORM"] = 'osmesa'

import numpy as np
import trimesh
import itertools
import sys
import argparse
import pyrender
from pyrender import (Scene, IntrinsicsCamera, Mesh,
                      Viewer, OffscreenRenderer, RenderFlags, Node)
from sd_maskrcnn.envs import CameraStateSpace

import matplotlib.pyplot as plt
import random
from termcolor import colored
import pickle
import torch
import cv2
from unsupervised_rbt.models import ResNetSiameseNetwork, InceptionSiameseNetwork
from unsupervised_rbt.losses.shapematch import ShapeMatchLoss
from pyquaternion import Quaternion
from tools.utils import *
import pickle
from tqdm import tqdm

def Matrix_Percent_Fit(mesh_dir, mesh_filename, obj_matrix, prism_matrix, mesh_expand):
    mesh = trimesh.load_mesh(os.path.join(mesh_dir, mesh_filename))
    obj_matrix[:3,3], prism_matrix[:3,3] = 0, 0
    prism_mesh = Aligned_Prism_Mesh(mesh,mesh_expand)
    mesh.apply_transform(obj_matrix)
    prism_mesh.apply_transform(prism_matrix)
    return Percent_Fit_Mesh(mesh, prism_mesh)
    
def Plot_Image(image, filename):
    """x
    """
    # cv2.imwrite(filename, image)
    fname2 = filename[:-4] + "_plt.png"
    fig1 = plt.imshow(image, cmap='gray', vmin=np.min(image[image != 0])-0.03)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname2)
    # plt.show()
    plt.close()

def parse_args():
    """Parse arguments from the command line.
    -config to input your own yaml config file. Default is data_gen_quat.yaml
    """
    parser = argparse.ArgumentParser()
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/data_gen_quat.yaml')
    parser.add_argument('--start', action='store_true')
    parser.add_argument('-config', type=str, default=default_config_filename)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = YamlConfig(args.config)

    mesh_expand = 1.1
    num_runs_per_obj = 100
    max_iterations = 5
    step_sizes = [0.8,0.8,0.8,0.8,0.8]
    network = 0
    # step_sizes = [1,0.8,0.6,0.4,0.2]

    asp = pickle.load(open("cfg/tools/data/eccentricities", "rb"))

    scene, renderer = create_scene()
    obj_id = -1
    light_pose = np.eye(4)
    light_pose[:3,3] = np.array([0.5,0.5,1])
    scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=light_pose)
    # light2 = np.eye(4)
    # light2[:3,3] = np.array([-0.5,0.5,1])
    # scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=light2)
    # light3 = np.eye(4)
    # light3[:3,3] = np.array([0.5,-0.5,1])
    # scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=light3)
    light4 = np.eye(4)
    light4[:3,3] = np.array([-0.5,-0.5,1])
    scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=light4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetSiameseNetwork(split_resnet=0).to(device)
    # print(torch.load("models/prismatic_cavity_large/cos_sm_blk0_reg9.pt").keys())
    model.load_state_dict(torch.load("models/prismatic_cavity_large/cos_sm_blk0_reg9.pt"))
    model.eval()
    pose_estim = PoseEstimator()

    losses, fits = [], []
    four_obj = [
                    ["/nfs/diskstation/objects/meshes/thingiverse", "rapsberry_pi_case_3218826.obj"],
                    ["/nfs/diskstation/objects/meshes/mini_dexnet", "endstop_holder.obj"],
                    ["/nfs/diskstation/objects/meshes/mini_dexnet", "part1.obj"],
                    ["/nfs/diskstation/objects/meshes/thingiverse", "curved_shield_3942823.obj"]
                ]
    # for mesh_dir, mesh_filename in Load_Mesh_Path():
    np.random.seed(62314)
    seeds = np.random.randint(0,9999999,num_runs_per_obj*4)
    for mesh_dir, mesh_filename in four_obj:
        obj_id += 1
        # if obj_id != 4:
        #     continue
        # object_list = [90,104,241,351] # 90 is twisty mug, 104 is golem, 241 is pharaoh, 351 is cat, 354 is chain mail
        # object_list = [49,90,104,124,184,185,241,304,344,351,359,382,384,406,414,528,537,555,595,639,665,731] #22 objects
        # object_list = np.loadtxt("cfg/tools/data/train_split_872")

        # if obj_id not in object_list:
        #     continue
        # if asp[obj_id] < 3 or asp[obj_id] > 9: # 9 objects within 2 and 9, 4 between 3 and 9: 351, 406, 528, 665
        #     continue

        # print(mesh_dir, mesh_filename)
        # print(colored('------------- Object ID ' + str(obj_id) + ' -------------', 'red'))

        # load object mesh
        mesh = Load_Scale_Mesh(mesh_dir, mesh_filename)

        obj_mesh = Mesh.from_trimesh(mesh)
        object_node = Node(mesh=obj_mesh, matrix=np.eye(4))

        prism_mesh = Mesh.from_trimesh(Aligned_Prism_Mesh(mesh,mesh_expand))
        prism_node = Node(mesh=prism_mesh, matrix=np.eye(4))
        scene.add_node(prism_node)

        base_path = "controller/objects/obj" + str(obj_id)

        losses_obj, fits_obj = [], []
        for j in tqdm(range(num_runs_per_obj)):
            np.random.seed(seeds[obj_id*num_runs_per_obj + j])
            matrices, quats = {}, {}
            goal_pose_matrix = Get_Initial_Pose(0.005,0.005,0.18,0.19)
            ctr_of_mass = goal_pose_matrix[0:3, 3]
            
            matrices["goal"], quats["goal"] = goal_pose_matrix, Rotation_to_Quaternion(goal_pose_matrix)

            # Render image 2, which will be the goal image of the bounding box
            scene.set_pose(prism_node, pose=goal_pose_matrix)
            # I_g = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
            I_g_rgb, I_g = renderer.render(scene)
            scene.remove_node(prism_node)
            scene.add_node(object_node)

            # Render image 1, which will be 30 degrees away from the goal
            rot_quat = Generate_Quaternion(29.9 /180 *np.pi,30.0 /180 *np.pi)

            # pred_quat = RigidTransform.quaternion_from_axis_angle([0,0,30.0/180*np.pi])
            # rot_quat = np.array([pred_quat[1],pred_quat[2],pred_quat[3],pred_quat[0]])
            # print(rot_quat, quats['goal'])

            start_pose_matrix = Quaternion_to_Rotation(rot_quat, ctr_of_mass) @ goal_pose_matrix #Doesn't matter that we reverse the operation from data gen because we only work with the absolute pose quaternions as in Save_Poses
            
            matrices[0], quats[0] = start_pose_matrix, Rotation_to_Quaternion(start_pose_matrix)

            scene.set_pose(object_node, pose=start_pose_matrix)
            # I_s = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
            I_s_rgb, I_s = renderer.render(scene)
            I_s, I_g = Quantize(Zero_BG(I_s)), Quantize(Zero_BG(I_g))

            Plot_Image(I_g_rgb, base_path + "/images/goal_rgb.png")
            Plot_Image(I_s_rgb, base_path + "/images/0_rgb.png")
            Plot_Image(I_g, base_path + "/images/goal.png")
            Plot_Image(I_s, base_path + "/images/0.png")

            im1_batch = torch.Tensor(torch.from_numpy(I_s).float()).to(device).unsqueeze(0).unsqueeze(0)
            im2_batch = torch.Tensor(torch.from_numpy(I_g).float()).to(device).unsqueeze(0).unsqueeze(0)

            cur_pose_matrix = matrices[0]
            total_iters = 1
            # for i in range(1,max_iterations+2): # +2 for CASE?
            for i in range(1,max_iterations+1):
                ctr_of_mass = cur_pose_matrix[:3,3]
                if network:
                    pred_quat = model(im1_batch,im2_batch).detach().cpu().numpy()[0]
                else:
                    pred_quat = pose_estim.get_rotation(im1_batch,im2_batch).detach().cpu().numpy()[0]
                    if i > 1:
                        break
                quat_reorder = np.array([pred_quat[3],pred_quat[0],pred_quat[1],pred_quat[2]])
                pred_quat = Quaternion(quat_reorder)
                rot_quat = Quaternion.slerp(Quaternion(), pred_quat, step_sizes[i-1]).elements # 1 is pred quat, 0 is no rot, used 0.2 for CASE
                rot_quat = np.array([rot_quat[1],rot_quat[2],rot_quat[3],rot_quat[0]])
                rot_quat = normalize(rot_quat)
                # print(rot_quat, np.linalg.norm(rot_quat))
                cur_pose_matrix = Quaternion_to_Rotation(rot_quat, ctr_of_mass) @ cur_pose_matrix

                matrices[i], quats[i] = cur_pose_matrix, Rotation_to_Quaternion(cur_pose_matrix)

                scene.set_pose(object_node, pose=cur_pose_matrix)
                cur_image = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
                cur_image = Quantize(Zero_BG(cur_image))

                if i < 20:
                    Plot_Image(cur_image, base_path + "/images/" + str(i) + ".png")
                im1_batch = torch.Tensor(torch.from_numpy(cur_image).float()).to(device).unsqueeze(0).unsqueeze(0)
                total_iters = i

                # if pred_quat[3] >= 0.9998476952: # 2 degrees
                # if pred_quat[3] >= 0.99996192306: # 1 degree
                # if pred_quat[3] >= np.cos(0.025/180*np.pi): # 0.5 degrees
                if pred_quat[3] >= 1: # 0 degrees
                    # print("stopping criteria:", np.arccos(pred_quat[3]) * 180 /np.pi*2, pred_quat)
                    break

            losses_run, fits_run = [], []
            quat_goal = quats["goal"]

            np.random.seed()
            for i in range(total_iters+1):
                q, m = quats[i], matrices[i]
                angle_error = np.arccos(np.abs(np.dot(quat_goal, q)))*180/np.pi*2
                fit = Matrix_Percent_Fit(mesh_dir, mesh_filename, m, matrices["goal"],mesh_expand)
                losses_run.append(angle_error)
                fits_run.append(fit)
            # print(np.round(losses_run,2)[::5])
            print(np.round(fits_run,2)) if num_runs_per_obj < 5 else 0
            
            plt.plot(losses_run)
            plt.title("Angle Difference Between Iteration Orientation and Goal Orientation")
            plt.ylabel("Angle Difference (Degrees)")
            plt.xlabel("Iteration Number")
            plt.savefig(base_path + "/images/loss.png")
            plt.close()
            losses_obj.append(losses_run)
            fits_obj.append(fits_run)
            scene.remove_node(object_node)
            scene.add_node(prism_node)
            # pickle.dump(quats, open(base_path + "/poses/quats_" + str(j), "wb"))
            # pickle.dump(matrices, open(base_path + "/poses/matrices_" + str(j), "wb"))
        # delete the object to make room for the next
        scene.remove_node(prism_node)
        losses.append(losses_obj)
        fits.append(fits_obj)
        np.save("controller/results/losses_ecc3", np.array(losses))
        np.save("controller/results/fits_ecc3", np.array(fits))
