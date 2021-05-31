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
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-config', type=str, default=default_config_filename)
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
    if not config['debug']:
        tensor_config = config['dataset']['tensors']
        dataset = TensorDataset("/nfs/diskstation/projects/unsupervised_rbt/" + name_gen_dataset + "/", tensor_config)
        datapoint = dataset.datapoint_template
    scene, renderer = create_scene_real(config)
    # print("NUM OBJECTS")
    # print([len(a) for a in mesh_lists])

    num_samples_per_obj = config['num_samples_per_obj']

    obj_id = 0
    data_point_counter = 0
    wrong_counter = 0
    # scales = pickle.load(open("cfg/tools/data/scales", "rb"))
    # print(max(scales.values()), min(scales.values()))
    # split = np.loadtxt('cfg/tools/data/train_split_546')

    objects_added, points_1000, eccentricities, max_bb = {}, {}, {}, {}
    scores = np.loadtxt("cfg/tools/data/final_scores")
    # split_872 = np.loadtxt("cfg/tools/data/train_split_872")

    light_pose = np.eye(4)
    light_pose[:3,3] = np.array([0.5,0.5,1])
    scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=light_pose)

    for mesh_dir, mesh_filename in Load_Mesh_Path():
        obj_id += 1
        # if obj_id != 73: #2 donut, 4 elephant, 6 is bottle, 31 L-Shaped, 73 L-Shaped, 90 twisty mug, 156 polygonal insertion
        #     continue
        # dataset.flush()
        # sys.exit(0)
        # if scores[obj_id-1] < 156.5:
        #     continue

        print(colored('------------- Object ID ' + str(obj_id) + ' -------------', 'red'))
        start_time = time.time()
        mesh = Load_Scale_Mesh(mesh_dir, mesh_filename,0.07,0.2) #CASE is 0.2 - 0.25

        # points_1000[obj_id] = Sample_Mesh_Points(mesh, vertices=False, n_points=1000)

        # trans, bounds = trimesh.bounds.oriented_bounds(mesh,5,ordered=False)
        # bound_ecc = np.max(bounds)/np.min(bounds)
        # extent_ecc = np.max(mesh.extents)/np.min(mesh.extents)
        # print("Bounds Ecc:", bound_ecc)
        # print("Extents Ecc:", extent_ecc)
        # if bound_ecc < extent_ecc:
        #     wrong_counter +=1
        # eccentricities[obj_id] = bound_ecc
        # max_bb[obj_id] = np.max(bounds)
        
        obj_mesh = Mesh.from_trimesh(mesh)
        object_node = Node(mesh=obj_mesh, matrix=np.eye(4))
        scene.add_node(object_node)

        j = 0
        while j < num_samples_per_obj:
            rand_transform = Get_Initial_Pose(0.05,0.07,0.2,0.4, rotation = "SO3")
            ctr_of_mass = rand_transform[0:3, 3]

            # Render image 1, which will be our original image with a random initial pose
            scene.set_pose(object_node, pose=rand_transform)
            image1 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
            # image1, depth_im = renderer.render(scene, RenderFlags.SHADOWS_DIRECTIONAL)

            # Render image 2, which will be image 1 rotated according to our specification
            random_quat = Generate_Quaternion(end = np.pi/6)
            quat_str = Quaternion_String(random_quat)

            new_pose = Quaternion_to_Rotation(random_quat, ctr_of_mass) @ rand_transform
            scene.set_pose(object_node, pose=new_pose)
            image2 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)

            original1, original2 = image1, image2
            Plot_Image(original1, "test.png") if config['debug'] else 0
            Plot_Image(original2, "test2.png")if config['debug'] else 0

            try:
                image1, image2 = Crop_Image(image1), Crop_Image(image2)
            except AssertionError:
                print("Bad location for object")
                continue
            image1 = Cut_Image(image1)
            image1, image2 = Zero_BG(image1), Zero_BG(image2)

            if config['debug']:
                print(image1[image1!=0].max(), image1.min())
                Plot_Datapoint(image1, image2, random_quat)
                sys.exit()

            datapoint = dataset.datapoint_template
            datapoint["depth_image1"] = np.expand_dims(image1, -1)
            datapoint["depth_image2"] = np.expand_dims(image2, -1)
            datapoint["quaternion"] = random_quat
            datapoint["lie"] = Quat_to_Lie(random_quat)
            datapoint["obj_id"] = obj_id
            datapoint["pose_matrix"] = rand_transform

            data_point_counter += 1
            dataset.add(datapoint)
            objects_added[obj_id] = 1
            
            if j % 10 == 9:
                scene.remove_node(object_node)
                mesh = Load_Scale_Mesh(mesh_dir, mesh_filename, 0.07, 0.2)
                obj_mesh = Mesh.from_trimesh(mesh)
                object_node = Node(mesh=obj_mesh, matrix=np.eye(4))
                scene.add_node(object_node)
            
            j += 1

        print("Added object", obj_id, "and overall datapoints are:", data_point_counter, 
                            "in", round(time.time() - start_time, 2), "seconds")
        # delete the object to make room for the next
        scene.remove_node(object_node)
    objects_added = np.array(list(objects_added.keys()),dtype=int)
    np.random.shuffle(objects_added)
    print("Added", data_point_counter, "datapoints to dataset from ", len(objects_added), "objects")
    
    # if num_samples_per_obj > 0:
        # print("Obj ID to split on training and validation:")
        # print(objects_added[:len(objects_added)//5])
    #     np.savetxt("cfg/tools/data/train_split872", objects_added[:len(objects_added)//5])
    #     np.savetxt("cfg/tools/data/test_split872", objects_added[len(objects_added)//5:])
    # pickle.dump(points_1000, open("cfg/tools/data/surface_pc_1000", "wb"))
    # print(wrong_counter, "Had base ecc higher than oriented ecc")
    # pickle.dump(eccentricities, open("cfg/tools/data/eccentricities", "wb"))
    # pickle.dump(max_bb, open("cfg/tools/data/max_bb", "wb"))
    dataset.flush()
