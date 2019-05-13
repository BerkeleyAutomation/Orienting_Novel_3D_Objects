from autolab_core import YamlConfig, RigidTransform, TensorDataset
import os
# os.environ["PYOPENGL_PLATFORM"] = 'osmesa'

import numpy as np
import trimesh
import itertools
import sys
import argparse

from pyrender import (Scene, PerspectiveCamera, Mesh, 
                      Viewer, OffscreenRenderer, RenderFlags, Node)   
from sd_maskrcnn.envs import CameraStateSpace

import matplotlib.pyplot as plt
import random
from termcolor import colored

def normalize(z):
    return z / np.linalg.norm(z)

def create_scene():
    scene = Scene()
    renderer = OffscreenRenderer(viewport_width=1, viewport_height=1)
    
    # initialize camera and renderer
    cam = CameraStateSpace(config['state_space']['camera']).sample()
    camera = PerspectiveCamera(cam.yfov, znear=0.05, zfar=3.0,
                                   aspectRatio=cam.aspect_ratio)
    renderer.viewport_width = cam.width
    renderer.viewport_height = cam.height
    
    pose_m = cam.pose.matrix.copy()
    pose_m[:,1:3] *= -1.0
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
    table_mesh = Mesh.from_trimesh(table_mesh)
    table_node = Node(mesh=table_mesh, matrix=table_tf.matrix)
    scene.add_node(table_node)

    # scene.add(Mesh.from_trimesh(table_mesh), pose=table_tf.matrix, name='table')
    return scene, renderer

def parse_args():
    parser = argparse.ArgumentParser()
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/data_gen_downstream.yaml')
    parser.add_argument('-config', type=str, default=default_config_filename)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = YamlConfig(args.config)

    # to adjust
    name_gen_dataset = "downstream"
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

    obj_id = 0
    for mesh_dir, mesh_list in zip(mesh_dir_list, mesh_lists):
        for mesh_filename in mesh_list:
            obj_id += 1
            if obj_id >= 50:
                dataset.flush()
                sys.exit(0)
            # log
            print(colored('------------- Object ID ' + str(obj_id) + ' -------------', 'red'))

            # load object mesh
            mesh = trimesh.load_mesh(os.path.join(mesh_dir, mesh_filename))
            obj_mesh = Mesh.from_trimesh(mesh)
            object_node = Node(mesh=obj_mesh, matrix=np.eye(4))
            scene.add_node(object_node)

            # calculate stable poses
            stable_poses, _ = mesh.compute_stable_poses(
                sigma=obj_config['stp_com_sigma'],
                n_samples=obj_config['stp_num_samples'],
                threshold=obj_config['stp_min_prob']
            )
            images = []
            for _ in range(config['num_samples_per_obj']):
                pose_id = np.random.choice(range(len(stable_poses)))
                pose_matrix = stable_poses[pose_id]
                ctr_of_mass = pose_matrix[0:3,3]
                rand_transform = RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=2*np.pi*np.random.random()).matrix @ pose_matrix
                scene.set_pose(object_node, pose=rand_transform)

                images.append(1 - renderer.render(scene, flags=RenderFlags.DEPTH_ONLY))

                # plt.imshow(images[-1], cmap='gray')
                # plt.show()

            datapoint = dataset.datapoint_template
            datapoint["depth_images"] = np.array(images)
            dataset.add(datapoint)
            scene.remove_node(object_node)
    dataset.flush()