'''
This script generates data for the self-supervised rotation prediction task
'''

from autolab_core import YamlConfig, RigidTransform, TensorDataset
from scipy.spatial.transform import Rotation
import os
import time
# Use this if you are SSH
os.environ["PYOPENGL_PLATFORM"] = 'egl'
# os.environ["PYOPENGL_PLATFORM"] = 'osmesa'

from dexgrasp import MeshLoader, YamlLoader
import torch
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
from tools.utils import *
from tools.stable_pose_utils import *
from tools.chamfer_distance import ChamferDistance
from tqdm import tqdm


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
                              znear = 0.4, zfar = 2)
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
    table_mesh.visual = table_mesh.visual.to_color()
    table_mesh.visual.vertex_colors = [[0 for c in r] for r in table_mesh.visual.vertex_colors]
    table_mesh = Mesh.from_trimesh(table_mesh)
    table_node = Node(mesh=table_mesh, matrix=table_tf.matrix)
    scene.add_node(table_node)

    # scene.add(Mesh.from_trimesh(table_mesh), pose=table_tf.matrix, name='table')
    return scene, renderer

def parse_args():
    """Parse arguments from the command line.
    -config to input your own yaml config file. Default is data_gen_quat.yaml
    -dataset to input a name for your dataset. Should start with quaternion
    --objpred to use the num_samples_per_obj_objpred option of your config
    """
    parser = argparse.ArgumentParser()
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/data_gen_quat.yaml')
    parser.add_argument('-config', type=str, default=default_config_filename)
    # parser.add_argument('-dataset', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    time.sleep(3)
    args = parse_args()
    config = YamlConfig(args.config)
    scene, renderer = create_scene()
    mesh_dataset_name_list = ['3dnet', 'thingiverse', 'kit']
    mesh_dir = config['state_space']['heap']['objects']['mesh_dir']
    mesh_dir_list = [os.path.join(mesh_dir, dataset_name) for dataset_name in mesh_dataset_name_list]
    obj_config = config['state_space']['heap']['objects']
    mesh_lists = [os.listdir(mesh_dir) for mesh_dir in mesh_dir_list]
    print("NUM OBJECTS")
    print([len(a) for a in mesh_lists])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_samples_per_obj = config['num_samples_per_obj']
    obj_id = 0

    scores = np.loadtxt("cfg/tools/data/final_scores")
    # split_872 = np.loadtxt("cfg/tools/data/train_split_872")
    for mesh_dir, mesh_list in zip(mesh_dir_list, mesh_lists):
        for mesh_filename in mesh_list:
            obj_id += 1
            if obj_id != 4: #4 elephant, 90 twisty mug, 2 donut
                continue
            # dataset.flush()
            # sys.exit(0)
            # if scores[obj_id-1] < 156.5:
            #     continue

            print(colored('------------- Object ID ' + str(obj_id) + ' -------------', 'red'))
            start_time = time.time()

            # load object mesh
            mesh = trimesh.load_mesh(os.path.join(mesh_dir, mesh_filename))
            obj_mesh_loader = MeshLoader(config['state_space']['heap']['objects']['mesh_dir'])
            adversarial_meshes = ["mini_dexnet~bar_clamp","mini_dexnet~vase", "mini_dexnet~endstop_holder", "mini_dexnet~pawn", "mini_dexnet~mount1", "mini_dexnet~pipe_connector", "mini_dexnet~gearbox"]
            mesh = obj_mesh_loader.load(adversarial_meshes[1])
            points = mesh.vertices
            if mesh.scale > 0.25:
                mesh.apply_transform(trimesh.transformations.scale_and_translate(0.25/mesh.scale)) #Final submission was 0.25
            if mesh.scale < 0.2:
                mesh.apply_transform(trimesh.transformations.scale_and_translate(0.2/mesh.scale)) #Final submission was 0.2

            obj_mesh = Mesh.from_trimesh(mesh)
            object_node = Node(mesh=obj_mesh, matrix=np.eye(4))
            scene.add_node(object_node)

            # light_pose = np.eye(4)
            # light_pose[:,3] = np.array([0.5,0.5,1,1])
            # scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=light_pose) # for rgb?

            # calculate stable poses
            stable_poses, stp_probs = mesh.compute_stable_poses(threshold=0.001)
            if len(stable_poses) == 0:
                print("No Stable Poses")
                scene.remove_node(object_node)
                continue

            prev_pcs, prev_stps, gt_stps, gt_map, next_stp = [], [], [], {}, 0

            for i in tqdm(range(num_samples_per_obj)):
            # for i in range(num_samples_per_obj):
                # sp1_idx = i % len(stable_poses) if i < 2 * len(stable_poses) else np.random.randint(0,len(stable_poses))
                sp1_idx = np.random.randint(0,len(stable_poses))
                sp1 = stable_poses[sp1_idx].copy()
                sp1[:2,3] += np.random.uniform(-0.02,0.02,2)
                sp1[2,3] += 0.2

                ctr_of_mass1 = sp1[0:3, 3]

                # rand_transform = np.eye(4) @ sp1
                rand_transform = Generate_Random_Z_Transform(ctr_of_mass1) @ sp1

                scene.set_pose(object_node, pose=rand_transform)
                image1 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)

                # rgb_im, depth_im = renderer.render(scene, RenderFlags.SHADOWS_DIRECTIONAL)
                image1 = Zero_BG(image1, DR=False)
                pc1 = torch.Tensor(demean(pointcloud(image1))).float().to(device)
                if config['debug']:
                    Plot_Image(image1, "stp" + str(sp1_idx) + ".png")
                    # Plot_PC([pc1.cpu().numpy().T], "plots/stp" + str(sp1_idx)+ "_pc.png")
                    # print(cur_pc.cpu().numpy().T)

                for j, prev_pc in enumerate(prev_pcs):
                    result = is_match(pc1, prev_pc)
                    if result[0]:
                        prev_stps.append(prev_stps[j])
                        prev_pcs.append(pc1)
                        break
                if len(prev_stps) == i:
                    prev_stps.append(next_stp)
                    prev_pcs.append(pc1)
                    next_stp += 1
                if sp1_idx not in gt_map.keys():
                    gt_map[sp1_idx] = max(gt_map.values()) + 1 if len(gt_map.values()) else 0
                gt_stps.append(gt_map[sp1_idx])
                # print("Iteration", i, "Stable Pose is", gt_map[sp1_idx], "Predicting to be", prev_stps[-1])

                # if not result[0]:
                #     sys.exit()
            print("Finished", num_samples_per_obj, "iterations in", round(time.time() - start_time, 2), "seconds")
            # results_arr = np.array(results)
            # print("Accuracy is", np.sum(results_arr[:,2]==results_arr[:,3])/results_arr.shape[0])
            # delete the object to make room for the next
            scene.remove_node(object_node)
    cm = confusion_matrix(gt_stps, prev_stps, normalize='true')
    ax = plt.figure(figsize=(9,6)).gca()
    disp = ConfusionMatrixDisplay(cm,range(max(gt_stps)+1))
    disp.plot(cmap='GnBu', ax = ax)
    plt.savefig("plots/class_confusion/vase.png")
    np.save("plots/class_confusion/vase.npy", np.array([list(gt_map.keys()),list(gt_map.values())]))
    print(np.array([list(gt_map.keys()),list(gt_map.values())]))
    plt.close()

