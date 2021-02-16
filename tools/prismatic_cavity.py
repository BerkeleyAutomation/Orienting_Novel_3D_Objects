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
    parser.add_argument('-dataset', type=str, default = "prismatic_cavity_test2")
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

    objects_added = {}
    scores = np.loadtxt("cfg/tools/data/final_scores")

    light_pose = np.eye(4)
    light_pose[:,3] = np.array([0.5,0.5,1,1])
    scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=light_pose) # for rgb?
    asp = pickle.load(open("cfg/tools/data/eccentricities", "rb"))

    for mesh_dir, mesh_filename in Load_Mesh_Path():
        obj_id += 1
        # if obj_id != 190: #2 donut, 4 elephant, 6 is bottle, 8 is screwdriver, 14 is cuboid, 90 twisty mug, 156 polygonal insertion, 190 is base pose ecc 1.3, min vol bb ecc 6.2
        #     continue
        # dataset.flush()
        # sys.exit(0)
        # if scores[obj_id-1] < 156.5:
        #     continue
        if asp[obj_id] < 2:
            continue

        print(colored('------------- Object ID ' + str(obj_id) + ' -------------', 'red'))
        start_time = time.time()

        mesh = Load_Scale_Mesh(mesh_dir,mesh_filename,0.17,0.21)
        
        # trans, bounds = trimesh.bounds.oriented_bounds(mesh,5,ordered=False)
        # bound_ecc = np.max(bounds)/np.min(bounds)
        # extent_ecc = np.max(mesh.extents)/np.min(mesh.extents)
        # print(bounds)
        # print("Bounds Ecc:", bound_ecc)
        # print(mesh.extents)
        # print("Extents Ecc:", extent_ecc)
        # if bound_ecc / extent_ecc > 1.5:
        #     sys.exit()

        obj_mesh = Mesh.from_trimesh(mesh)
        object_node = Node(mesh=obj_mesh, matrix=np.eye(4))
        scene.add_node(object_node)

        prism_mesh = Aligned_Prism_Mesh(mesh, expand=1)
        prism_mesh_pyrender = Mesh.from_trimesh(prism_mesh) #SHOULD USE ORIENTED BB
        prism_node = Node(mesh=prism_mesh_pyrender, matrix=np.eye(4))

        # sampled_points = trimesh.sample.volume_mesh(mesh, 1000)
        # n= sampled_points.shape[0]
        # print(np.sum(prism_mesh.contains(sampled_points))/n)
        # print(np.sum(prism_mesh2.contains(sampled_points))/n)

        for j in range(num_samples_per_obj):
            rand_transform = Get_Initial_Pose(0.005,0.18,0.19) #0.01 CASE #0.18-0.21 CASE
            # rand_transform[:3,:3] = np.eye(3) #TODO CHANGE
            ctr_of_mass = rand_transform[0:3, 3] 

            scene.set_pose(object_node, pose=rand_transform)
            image1 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)

            scene.set_pose(object_node, pose=np.eye(4))
            scene.remove_node(object_node)

            # Render image 2, which will be image 1 rotated according to our specification
            random_quat = Generate_Quaternion(end = np.pi/6)
            quat_str = Quaternion_String(random_quat)
            new_pose = Quaternion_to_Rotation(random_quat, ctr_of_mass) @ rand_transform

            scene.add_node(prism_node)
            scene.set_pose(prism_node, pose=new_pose)
            image2 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)

            scene.remove_node(prism_node)
            scene.add_node(object_node)

            # image1 = Cut_Image(image1)
            image1, image2 = Zero_BG(image1), Zero_BG(image2)

            datapoint = dataset.datapoint_template
            datapoint["depth_image1"] = np.expand_dims(image1, -1)
            datapoint["depth_image2"] = np.expand_dims(image2, -1)
            datapoint["quaternion"] = random_quat
            datapoint["lie"] = Quat_to_Lie(random_quat)
            datapoint["obj_id"] = obj_id
            datapoint["pose_matrix"] = rand_transform

            if config['debug']:
                print(image1[image1 != 0].min(), image1.max())
                print(image2[image2 != 0].min(), image2.max())
                Plot_Image(image1, "test.png")
                Plot_Image(image2, "test2.png")
                Plot_Datapoint(image1, image2, random_quat)
                sys.exit()
            data_point_counter += 1
            dataset.add(datapoint)
            objects_added[obj_id] = 1

        print("Added object", obj_id, "and overall datapoints are:", data_point_counter, 
                            "in", round(time.time() - start_time, 2), "seconds")
        # delete the object to make room for the next
        scene.remove_node(object_node)
    objects_added = np.array(list(objects_added.keys()),dtype=int)
    np.random.shuffle(objects_added)
    print("Added", data_point_counter, "datapoints to dataset from ", len(objects_added), "objects")
    dataset.flush()
