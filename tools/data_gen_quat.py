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
from tools.utils import *

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
                              cam.intrinsics.cx, cam.intrinsics.cy)
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
    -dataset to input a name for your dataset. Should start with quaternion
    --objpred to use the num_samples_per_obj_objpred option of your config
    """
    parser = argparse.ArgumentParser()
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/data_gen_quat.yaml')
    parser.add_argument('-config', type=str, default=default_config_filename)
    parser.add_argument('-dataset', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = YamlConfig(args.config)
    # to adjust
    name_gen_dataset = args.dataset
    if config['debug']:
        name_gen_dataset += "_junk"

    # dataset configuration
    tensor_config = config['dataset']['tensors']
    dataset = TensorDataset("/nfs/diskstation/projects/unsupervised_rbt/" + name_gen_dataset + "/", tensor_config)
    datapoint = dataset.datapoint_template

    scene, renderer = create_scene()
    dataset_name_list = ['3dnet', 'thingiverse', 'kit']
    mesh_dir = config['state_space']['heap']['objects']['mesh_dir']
    mesh_dir_list = [os.path.join(mesh_dir, dataset_name) for dataset_name in dataset_name_list]
    obj_config = config['state_space']['heap']['objects']
    mesh_lists = [os.listdir(mesh_dir) for mesh_dir in mesh_dir_list]
    print("NUM OBJECTS")
    print([len(a) for a in mesh_lists])

    num_samples_per_obj = config['num_samples_per_obj']

    obj_id = 0
    data_point_counter = 0

    # split = np.loadtxt('cfg/tools/train_split_546')

    symmetries = [
        3, 3, 3, 0, 0, 3, 0.5, 1.5, 1, 3, 0, 1, 2, 3, 1, 0, 1, 3, 2, 2, 3, 3, 3, 3, 1, 0, 0, 1.5, 2, 2, 2, 3, 1, 1, 3, 0, 3, 1, 1, 2, 0.5, 3, 0, 1.5,
        1, 1, 3, 1, 0, 1.5, 2.5, 1, 0, 3, 3, 0, 3, 3, 1.5, 2, 3, 1.5, 3, 2, 3, 1.5, 3, 1, 2, 3, 3, 2, 1, 3, 1, 3, 0.5, 2, 1, 0.5, 1.5, 3, 2, 2, 2, 2,
        2, 3, 2, 2, 3, 0, 3, 3, 3, 2, 0, 1, 2, 3, 3, 3, 3, 0, 3, 0, 3, 3, 3, 2, 1.5, 3, 1, 3, 3, 3, 3, 3, 0, 3, 0, 3, 1, 0, 3, 3, 2, 0, 3, 3, 3, 2, 1,
        3, 1.5, 3, 3, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0.5, 0, 2, 3, 2, 3, 3, 2, 3, 2, 2, 3, 2, 1.5, 3, 0.5, 3, 404, 2, 1.5, 2, 3, 3, 3, 3, 3, 3, 2,
        2, 3, 2, 3, 2, 2, 404, 2, 2, 2, 3, 2, 404, 3, 1.5, 0, 2, 0, 0, 2, 2, 2, 0, 0, 1, 1.5, 2, 2, 0, 1, 2, 2, 3, 3, 0, 2, 0.5, 3, 3, 3, 404, 3, 3,
        3, 3, 2, 2, 3, 0.5, 404, 2, 1, 1.5, 2.5, 0, 3, 1.5, 0, 0, 2.5, 3, 1.5, 666, 3, 0.5, 3, 3, 0, 3, 404, 3, 3, 1.5, 3, 2.5, 2, 2, 3, 3, 1, 2, 2,
        2, 999, 0.5, 3, 3, 3, 3, 0, 3, 0, 0.5, 2, 0.5, 0.5, 2, 2, 2.5, 3, 3, 1, 2, 1, 1.5, 3, 2, 2, 3, 2, 1.5, 0.5, 0.5, 1, 3, 3, 0, 3, 1, 1, 1, 2.5,
        3, 3, 3, 3, 2, 0, 3, 3, 3, 2, 404, 666, 3, 2, 2, 2, 3, 3, 3, 1.5, 0, 2, 2, 2, 2, 3, 3, 3, 0, 3, 3, 3, 2, 0.5, 3, 3, 3, 0.5, 404, 2, 3, 0.5, 3,
        3, 3, 0, 3, 1, 3, 3, 3, 3, 0, 3, 404, 0.5, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2.5, 2, 3, 0, 2.5, 0, 2, 0.5, 0.5, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 666, 0, 3, 1, 0, 0.5, 0, 2, 2, 3, 0, 3, 2, 1, 3, 3, 3, 3, 404, 3, 3, 2, 3, 3, 1, 404, 3, 3, 0, 2, 1.5, 2, 404, 3, 2,
        2, 2, 2, 0, 0, 0, 0, 1, 0, 1.5, 0, 3, 3, 3, 3, 3, 2, 3, 1.5, 0, 1.5, 3, 3, 2, 1, 2, 1.5, 0, 2, 1.5, 2, 0, 3, 3, 0, 1.5, 0, 3, 3, 1.5, 1.5, 3,
        3, 3, 3, 2, 3, 3, 3, 3, 1.5, 0, 3, 3, 3, 3, 3, 404, 3, 2, 2, 2, 1, 1, 3, 3, 0, 2, 1.5, 1.5, 1.5, 1.5, 2, 3, 3, 666, 404, 3, 3, 3, 0.5, 1.5,
        1.5, 0, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 0, 3, 2, 2, 3, 2, 2, 1, 1.5, 3, 3, 1.5, 1.5, 2, 0, 3, 2, 2, 2, 3, 1, 1.5, 1.5, 1.5, 3, 1.5, 1.5, 3,
        2, 1.5, 3, 3, 2, 1.5, 3, 3, 3, 404, 3, 0, 3, 2, 3, 3, 3, 2, 2, 404, 3, 2, 3, 3, 3, 2, 2, 1.5, 3, 2, 3, 3, 3, 3, 3, 0, 2, 2, 3, 3, 3, 3, 1, 0,
        0, 0, 3, 1.5, 3, 1.5, 3, 3, 1.5, 1.5, 3, 2, 1.5, 0.5, 3, 3, 2, 3, 2, 3, 2, 1.5, 0, 1.5, 0, 3, 2, 2, 1, 0, 3, 0, 1.5, 3, 3, 3, 3, 3, 2, 1.5, 3,
        3, 3, 3, 404, 3, 2, 1.5, 3, 3, 1, 3, 3, 3, 3, 1.5, 3]
    best_obj = [index +1 for index, value in enumerate(symmetries) if value == 0]
    best_obj_scores = [327, 423, 490, 555, 438, 421, 496, 553, 113, 566, 272, 310, 150, 304, 462, 346, 556, 243, 4, 530, 427, 228, 313, 592, 639, 244, 359,
                594, 608, 763, 16, 660, 13, 731, 634, 205, 491, 523, 621, 466, 104, 256, 26, 382, 340, 834, 260, 227, 653, 464, 89, 737, 596, 306,
                766, 399, 231, 177, 544, 726, 353, 83, 184, 655, 455, 230, 351, 650, 90, 235]
    # best_obj_scores += [5]
    dont_include = [555,310,304, 462, 243, 228, 313,592, 359, 763, 13, 634, 491, 621,466, 340, 227,653,464,89,596,306,177,353,83,184,230,650,90]
    objects_added = {}
    all_points, all_points300, all_scales = {}, {}, {}
    scores = np.loadtxt("cfg/tools/final_scores")
    for mesh_dir, mesh_list in zip(mesh_dir_list, mesh_lists):
        for mesh_filename in mesh_list:
            obj_id += 1
            # if obj_id != 4:
            #     continue
            # if obj_id > 20:
            #     break
            # dataset.flush()
            # sys.exit(0)
            # if obj_id > len(symmetries) or obj_id not in best_obj: # or symmetries[obj_id-1] != 0:
            #     continue
            # if (obj_id not in best_obj_scores and obj_id not in best_obj) or obj_id in dont_include: # or symmetries[obj_id-1] != 0:
            #     continue
            # if obj_id not in best_obj_scores or obj_id in dont_include: # or symmetries[obj_id-1] != 0:
            #     continue
            if scores[obj_id-1] < 156.5:
                continue

            print(colored('------------- Object ID ' + str(obj_id) + ' -------------', 'red'))

            # load object mesh
            mesh = trimesh.load_mesh(os.path.join(mesh_dir, mesh_filename))
            points = mesh.vertices
            if points.shape[0] < 300:
                continue
            # print(points.shape)
            # all_points[obj_id] = points.T
            # all_scales[obj_id] = mesh.scale
            # if points.shape[0] >= 300:
            #     points_clone = np.copy(points)
            #     np.random.shuffle(points_clone)
            #     all_points300[obj_id] = points_clone[:300].T

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

            for _ in range(max(num_samples_per_obj // len(stable_poses), 1)):
                # iterate over all stable poses of the object
                for j, pose_matrix in enumerate(stable_poses):
                    # print(pose_matrix[:,3])
                    pose_matrix = pose_matrix.copy()
                    pose_matrix[:2,3] += np.random.uniform(-0.02,0.02,2)
                    pose_matrix[2,3] += np.random.uniform(0,0.2)
                    # print(pose_matrix[:,3])

                    ctr_of_mass = pose_matrix[0:3, 3]

                    # Render image 1, which will be our original image with a random initial pose
                    # rand_transform = Generate_Random_Z_Transform(ctr_of_mass) @ pose_matrix
                    # rand_transform = Generate_Random_Transform(ctr_of_mass) @ pose_matrix
                    rand_transform = Generate_Random_TransformSO3(ctr_of_mass) @ pose_matrix
                    scene.set_pose(object_node, pose=rand_transform)
                    image1 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)

                    # if j == 0:
                    #     image1, depth_im = renderer.render(scene, RenderFlags.SHADOWS_DIRECTIONAL)
                    #     fig1 = plt.imshow(image1)
                    #     fig1.axes.get_xaxis().set_visible(False)
                    #     fig1.axes.get_yaxis().set_visible(False)
                    #     plt.show()
                    #     if obj_id in split:
                    #         plt.savefig("pictures/rgb_images/symmetric546/test/obj" + str(obj_id) + ".png")
                    #     else:
                    #         plt.savefig("pictures/rgb_images/symmetric546/train/obj" + str(obj_id) + ".png")
                    #     plt.close()

                    # Render image 2, which will be image 1 rotated according to our specification
                    random_quat = Generate_Quaternion()
                    quat_str = Quaternion_String(random_quat)
                    new_pose = Quaternion_to_Rotation(random_quat, ctr_of_mass) @ rand_transform

                    scene.set_pose(object_node, pose=new_pose)
                    image2 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)

                    image_cut = image1

                    #Generate cuts
                    segmask_size = np.sum(image1 <= 1 - 0.20001)
                    grip = [0,0]
                    while image1[grip[0]][grip[1]] > 1-0.20001:
                        grip = np.random.randint(0,128,2)
                    iteration, threshold = 0, 0.7
                    while True:
                        slope = np.random.uniform(-1,1,2)
                        slope = slope[1]/np.max([slope[0], 1e-8])
                        xx, yy = np.meshgrid(np.arange(0,128), np.arange(0,128))
                        mask = (np.abs(yy-grip[1] - slope*(xx-grip[0])) <= (4/0.7*threshold)*(np.abs(slope)+1))
                        image_cut = image1.copy()
                        image_cut[mask] = np.max(image1)
                        # print(slope)
                        # plt.imshow(image_cut, cmap='gray')
                        # fig1.axes.get_xaxis().set_visible(False)
                        # fig1.axes.get_yaxis().set_visible(False)
                        # plt.show()
                        # plt.savefig("pictures/com_test/obj" + str(obj_id) + "segmask.png")
                        # plt.close()
                        if iteration % 1000 == 999:
                            threshold -= 0.05
                        if np.sum(image_cut <= 1 - 0.20001) >= 0.7 * segmask_size:
                            # print(np.sum(image_cut >= 0.200001), segmask_size)
                            break
                        iteration += 1
                    # mse = np.linalg.norm(image1 - image2)
                    # image_cut, image2 = addNoise(image_cut, config['noise']), addNoise(image2, config['noise'])
                    datapoint = dataset.datapoint_template
                    datapoint["depth_image1"] = np.expand_dims(image_cut, -1)
                    datapoint["depth_image2"] = np.expand_dims(image2, -1)
                    datapoint["quaternion"] = random_quat
                    datapoint["obj_id"] = obj_id
                    datapoint["pose_matrix"] = rand_transform

                    if config['debug']:
                        Plot_Datapoint(datapoint)
                    data_point_counter += 1
                    dataset.add(datapoint)
                    objects_added[obj_id] = 1

            print("Added object ", obj_id, " and overall datapoints are: ", data_point_counter)
            # delete the object to make room for the next
            scene.remove_node(object_node)
    objects_added = np.array(list(objects_added.keys()),dtype=int)
    np.random.shuffle(objects_added)
    print("Added ", data_point_counter, " datapoints to dataset from ", len(objects_added), "objects")
    print("Obj ID to split on training and validation:")
    print(objects_added[:len(objects_added)//5])
    # if num_samples_per_obj > 20:
    #     np.savetxt("cfg/tools/train_split", objects_added[:len(objects_added)//5])
    #     np.savetxt("cfg/tools/test_split", objects_added[len(objects_added)//5:])
    # print(all_points)
    # pickle.dump(all_points, open("cfg/tools/point_clouds", "wb"))
    # pickle.dump(all_points300, open("cfg/tools/point_clouds300", "wb"))
    # pickle.dump(all_scales, open("cfg/tools/scales", "wb"))
    dataset.flush()
