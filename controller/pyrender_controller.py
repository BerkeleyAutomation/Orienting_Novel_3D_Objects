'''
This script generates data for the self-supervised rotation prediction task
'''

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

def Plot_Image(image, filename):
    """x
    """
    # cv2.imwrite(filename, image)
    fname2 = filename[:-4] + "_plt.png"
    fig1 = plt.imshow(image, cmap='gray')
    fig1.axes.get_xaxis().set_visible(False)
    fig1.axes.get_yaxis().set_visible(False)
    plt.show()
    plt.savefig(fname2)
    plt.close()

def Save_Poses(pose_matrix ,index):
    """x
    """
    np.savetxt(base_path + "/poses/matrix_" + index + ".txt", pose_matrix)
    pose_quat = Rotation_to_Quaternion(pose_matrix)
    np.savetxt(base_path + "/poses/quat_" + index + ".txt", pose_quat)

def create_scene(data_gen=True):
    """Create scene for taking depth images.
    """
    scene = Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
    renderer = OffscreenRenderer(viewport_width=1, viewport_height=1)
    if not data_gen:
        config2 = YamlConfig(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                                          'cfg/tools/data_gen_quat.yaml'))
    else:
        config2 = config
    # initialize camera and renderer
    cam = CameraStateSpace(config2['state_space']['camera']).sample()

    # If using older version of sd-maskrcnn
    # camera = PerspectiveCamera(cam.yfov, znear=0.05, zfar=3.0,
    #                                aspectRatio=cam.aspect_ratio)
    camera = IntrinsicsCamera(cam.intrinsics.fx, cam.intrinsics.fy,
                              cam.intrinsics.cx, cam.intrinsics.cy,
                              znear=0.4, zfar=2)
    renderer.viewport_width = cam.width
    renderer.viewport_height = cam.height

    pose_m = cam.pose.matrix.copy()
    pose_m[:, 1:3] *= -1.0
    scene.add(camera, pose=pose_m, name=cam.frame)
    scene.main_camera_node = next(iter(scene.get_nodes(name=cam.frame)))

    # Add Table
    table_mesh = trimesh.load_mesh(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../data/objects/plane/plane.obj',
        )
    )
    table_tf = RigidTransform.load(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../data/objects/plane/pose.tf',
        )
    )
    table_mesh.visual.vertex_colors = [[0 for c in r] for r in table_mesh.visual.vertex_colors]
    table_mesh = Mesh.from_trimesh(table_mesh)
    table_node = Node(mesh=table_mesh, matrix=table_tf.matrix)
    scene.add_node(table_node)

    # scene.add(Mesh.from_trimesh(table_mesh), pose=table_tf.matrix, name='table')
    return scene, renderer

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

if __name__ == "__main__":
    args = parse_args()
    config = YamlConfig(args.config)

    # dataset configuration
    tensor_config = config['dataset']['tensors']

    scene, renderer = create_scene()
    dataset_name_list = ['3dnet', 'thingiverse', 'kit']
    mesh_dir = config['state_space']['heap']['objects']['mesh_dir']
    mesh_dir_list = [os.path.join(mesh_dir, dataset_name) for dataset_name in dataset_name_list]
    obj_config = config['state_space']['heap']['objects']
    mesh_lists = [os.listdir(mesh_dir) for mesh_dir in mesh_dir_list]

    obj_id = 0

    losses = []
    for mesh_dir, mesh_list in zip(mesh_dir_list, mesh_lists):
        for mesh_filename in mesh_list:
            obj_id += 1
            # if obj_id != 4:
            #     continue
            #90 is twisty mug, 104 is golem, 241 is pharaoh, 351 is animal, 354 is chain mail
            object_list = [90, 104, 241,351, 354, 304, 384,528,406,537,639,124,731,665,
                            555,184,49,595,382,359,185,344] # 22 objects, 354 is not in best_scores 82
            if obj_id not in object_list:
                continue
            # if obj_id > 20:
            #     break
            # sys.exit(0)

            print(colored('------------- Object ID ' + str(obj_id) + ' -------------', 'red'))

            # load object mesh
            mesh = trimesh.load_mesh(os.path.join(mesh_dir, mesh_filename))
            points = mesh.vertices

            obj_mesh = Mesh.from_trimesh(mesh)
            object_node = Node(mesh=obj_mesh, matrix=np.eye(4))
            scene.add_node(object_node)

            # light_pose = np.eye(4)
            # light_pose[:,3] = np.array([0.5,0.5,1,1])
            # scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=light_pose) # for rgb?

            # calculate stable poses
            stable_poses, _ = mesh.compute_stable_poses(
                sigma=obj_config['stp_com_sigma'],
                n_samples=obj_config['stp_num_samples'],
                threshold=obj_config['stp_min_prob']
            )

            if len(stable_poses) == 0:
                print("No Stable Poses")
                scene.remove_node(object_node)
                continue
            
            base_path = "controller/objects/obj" + str(obj_id)

            stbl_pose = stable_poses[0]
            center_of_mass_stbl = stbl_pose[:3,3]
            losses_obj = []

            num_runs_per_obj = 100
            max_iterations = 100

            for j in range(num_runs_per_obj):
                goal_pose_matrix = Generate_Random_Transform(center_of_mass_stbl) @ stbl_pose.copy()
                ctr_of_mass = goal_pose_matrix[0:3, 3]
                
                Save_Poses(goal_pose_matrix, "goal")

                # Render image 2, which will be the goal image of the object in a stable pose
                scene.set_pose(object_node, pose=goal_pose_matrix)
                image2 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
                # image2 = (image2*65535).astype(int)
                Plot_Image(image2, base_path + "/images/goal.png")

                # Render image 1, which will be 30 degrees away from the goal
                rot_quat = Generate_Quaternion(29.9 /180 *np.pi,np.pi/6)
                start_pose_matrix = Quaternion_to_Rotation(rot_quat, ctr_of_mass) @ goal_pose_matrix
                Save_Poses(start_pose_matrix, "0")

                scene.set_pose(object_node, pose=start_pose_matrix)
                image1 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
                # image1 = (image1*65535).astype(int)
                Plot_Image(image1, base_path + "/images/0.png")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = ResNetSiameseNetwork(4, n_blocks=1, embed_dim=1024, dropout=6).to(device)
                model.load_state_dict(torch.load("models/546objv5/cos_sm_blk1_emb1024_reg7_drop4.pt"))
                # model.load_state_dict(torch.load("models/546objv3/cos_sm_blk4_emb1024_reg7_drop4.pt"))
                model.eval()

                I_s = image1
                # print(I_s.shape, I_s.min(), I_s.max())
                I_g = image2
                im1_batch = torch.Tensor(torch.from_numpy(I_s).float()).to(device).unsqueeze(0).unsqueeze(0)
                im2_batch = torch.Tensor(torch.from_numpy(I_g).float()).to(device).unsqueeze(0).unsqueeze(0)
                # print(im1_batch.size())
                cur_pose_matrix = np.loadtxt(base_path +"/poses/matrix_0.txt")
                total_iters = 1
                for i in range(1,max_iterations):
                    ctr_of_mass = cur_pose_matrix[:3,3]
                    pred_quat = model(im1_batch,im2_batch).detach().cpu().numpy()[0]
                    # if pred_quat[3] >= 0.99996192306: # 1 degree
                    # if pred_quat[3] >= 0.9998476952: # 2 degrees
                    # if pred_quat[3] >= np.cos(0.025/180*np.pi): # 0.5 degrees
                    if pred_quat[3] >= 1: # 0 degrees
                        # print("stopping criteria:", np.arccos(pred_quat[3]) * 180 /np.pi*2, pred_quat)
                        break
                    quat_reorder = np.array([pred_quat[3],pred_quat[0],pred_quat[1],pred_quat[2]])
                    pred_quat = Quaternion(quat_reorder)
                    rot_quat = Quaternion.slerp(Quaternion(), pred_quat, 0.2).elements # 1 is pred quat, 0 is no rot
                    rot_quat = np.array([rot_quat[1],rot_quat[2],rot_quat[3],rot_quat[0]])
                    rot_quat = normalize(rot_quat)
                    # print(rot_quat, np.linalg.norm(rot_quat))
                    cur_pose_matrix = Quaternion_to_Rotation(rot_quat, ctr_of_mass) @ cur_pose_matrix
                    Save_Poses(cur_pose_matrix, str(i))
                    scene.set_pose(object_node, pose=cur_pose_matrix)
                    cur_image = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
                    # cur_image = (cur_image*65535).astype(int)
                    if i < 20:
                        Plot_Image(cur_image, base_path + "/images/" + str(i) + ".png")
                    im1_batch = torch.Tensor(torch.from_numpy(cur_image).float()).to(device).unsqueeze(0).unsqueeze(0)
                    total_iters = i
                losses_j = []
                quat_goal = np.loadtxt(base_path + "/poses/quat_goal.txt")
                # print(quat_goal)
                for i in range(total_iters):
                    q = np.loadtxt(base_path + "/poses/quat_" + str(i) + ".txt")
                    # print(q)
                    # print(np.dot(q,quat_goal))
                    angle_error = np.arccos(np.abs(np.dot(quat_goal, q)))*180/np.pi*2
                    losses_j.append(angle_error)
                # print(np.round(losses_j,2)[::5])
                plt.plot(losses_j)
                plt.title("Angle Difference Between Iteration Orientation and Goal Orientation")
                plt.ylabel("Angle Difference (Degrees)")
                plt.xlabel("Iteration Number")
                plt.savefig(base_path + "/images/loss.png")
                plt.close()
                losses_obj.append(losses_j)
            # delete the object to make room for the next
            scene.remove_node(object_node)
            losses.append(losses_obj)
    print("Saving results to file")
    np.save("controller/results/losses", np.array(losses))

    # pickle.dump(all_points, open("cfg/tools/point_clouds", "wb"))
    # pickle.dump(all_points300, open("cfg/tools/point_clouds300", "wb"))
    # pickle.dump(all_scales, open("cfg/tools/scales", "wb"))
