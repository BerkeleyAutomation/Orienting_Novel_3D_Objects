from autolab_core import YamlConfig, RigidTransform, TensorDataset
from scipy.spatial.transform import Rotation
import os
import time

from torch import rand
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
    scene2 = Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
    renderer = OffscreenRenderer(viewport_width=1, viewport_height=1)
    if not data_gen:
        config2 = YamlConfig(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                                          'cfg/tools/data_gen_quat.yaml'))
    else:
        config2 = config
    # initialize camera and renderer
    cam = CameraStateSpace(config2['state_space']['camera']).sample()

    camera = IntrinsicsCamera(cam.intrinsics.fx, cam.intrinsics.fy,
                              cam.intrinsics.cx, cam.intrinsics.cy,
                              znear = 0.4, zfar = 2)
    renderer.viewport_width = cam.width
    renderer.viewport_height = cam.height

    pose_m = cam.pose.matrix.copy()
    pose_m[:, 1:3] *= -1.0
    scene.add(camera, pose=pose_m, name=cam.frame)
    scene.main_camera_node = next(iter(scene.get_nodes(name=cam.frame)))
    scene2.add(camera, pose=pose_m, name=cam.frame)
    scene2.main_camera_node = next(iter(scene2.get_nodes(name=cam.frame)))

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
    table_mesh_pyrender = Mesh.from_trimesh(table_mesh)
    table_node = Node(mesh=table_mesh_pyrender, matrix=table_tf.matrix)
    scene.add_node(table_node)
    lower_table_tf = table_tf.matrix.copy()
    lower_table_tf[:3,3] += np.array([0,0,-0.4])
    table_node2 = Node(mesh=table_mesh_pyrender, matrix=lower_table_tf)
    scene2.add_node(table_node2)

    # scene.add(Mesh.from_trimesh(table_mesh), pose=table_tf.matrix, name='table')
    return scene, renderer, table_mesh, table_node, scene2

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
    parser.add_argument('-dataset', type=str, default = "insertion_test")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = YamlConfig(args.config)
    # to adjust
    name_gen_dataset = args.dataset
    if config['debug']:
        name_gen_dataset += "_junk"
    else:
        # dataset configuration
        tensor_config = config['dataset']['tensors']
        dataset = TensorDataset("/nfs/diskstation/projects/unsupervised_rbt/" + name_gen_dataset + "/", tensor_config)
        datapoint = dataset.datapoint_template
    scene, renderer, table_mesh, table_node, scene2 = create_scene()


    table_mesh = trimesh.creation.box(extents = (1,1,1))
    table_trans = np.eye(4)
    table_trans[2,3] -= np.max(table_mesh.vertices,0)[2]
    table_mesh.apply_transform(table_trans)
    print(table_mesh.vertices.min(0),table_mesh.vertices.max(0))
    # table_mesh.export("plots/table_mesh.obj")

    dataset_name_list = ['3dnet', 'thingiverse', 'kit']
    mesh_dir = config['state_space']['heap']['objects']['mesh_dir']
    mesh_dir_list = [os.path.join(mesh_dir, dataset_name) for dataset_name in dataset_name_list]
    obj_config = config['state_space']['heap']['objects']
    mesh_lists = [os.listdir(mesh_dir) for mesh_dir in mesh_dir_list]
    print("NUM OBJECTS")
    print([len(a) for a in mesh_lists])

    num_samples_per_obj = config['num_samples_per_obj']

    obj_id, data_point_counter = 0, 0
    objects_added = {}
    light_pose = np.eye(4)
    light_pose[:,3] = np.array([0.5,0.5,1,1])
    scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=light_pose) # for rgb?
    scene2.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=light_pose) # for rgb?

    for mesh_dir, mesh_filename in Load_Mesh_Path():
        obj_id += 1
        if obj_id != 13: #2 donut, 4 elephant, 6 is bottle, 13 regular mug, 90 twisty mug, 156 polygonal insertion, 6 is bottle
            continue

        print(colored('------------- Object ID ' + str(obj_id) + ' -------------', 'red'))
        start_time = time.time()

        # load object mesh
        mesh = Load_Scale_Mesh(mesh_dir, mesh_filename, 0.1, 0.9)
        mesh = mesh.process(validate=True)
        table_mesh = table_mesh.process(validate=True)
        for j in range(num_samples_per_obj):
            pose_matrix = np.eye(4)
            # pose_matrix[:2,3] += np.random.uniform(-0.01,0.01,2)
            # pose_matrix[2,3] += np.random.uniform(0.18,0.23)

            pose_matrix[2,3] += np.random.uniform(0.035,0.0351)
            ctr_of_mass = pose_matrix[0:3, 3]

            # Render image 1, which will be our original image with a random initial pose
            # rand_transform = Get_Initial_Pose(0,0.035,0.0351)
            rand_transform = RigidTransform.rotation_from_axis_and_origin(axis = [0,1,0], 
                                            origin=ctr_of_mass , angle = np.pi/2).matrix @ pose_matrix
            obj_mesh = mesh.copy()
            obj_mesh.apply_transform(rand_transform)

            reflect_obj_mesh = obj_mesh.copy()
            # reflection = trimesh.transformations.reflection_matrix(ctr_of_mass, np.array([0,0,1]))
            # reflect_obj_mesh.apply_transform(reflection)

            reflect_obj_mesh_pyrender = Mesh.from_trimesh(reflect_obj_mesh)
            reflect_object_node = Node(mesh=reflect_obj_mesh_pyrender, matrix=np.eye(4))

            scene.add_node(reflect_object_node)

            # scene.set_pose(reflect_object_node, pose=rand_transform)
            # image1 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
            image1 = renderer.render(scene)[0]
            scene.remove_node(reflect_object_node)

            slice_list, num_slices = [table_mesh], 20
            random_quat = Generate_Quaternion(end = np.pi/180)
            quat_str = Quaternion_String(random_quat)
            new_pose = Quaternion_to_Rotation(random_quat, ctr_of_mass) @ rand_transform
            # obj_mesh.apply_transform(new_pose)
            center_trans = np.eye(4)
            center_trans[:3,3] -= ctr_of_mass
            center_trans[2,3] -= np.min(obj_mesh.vertices,0)[2]
            obj_mesh.apply_transform(center_trans)

            # new_pose = RigidTransform.rotation_from_axis_and_origin(axis = [0,1,0], 
            #                                 origin=ctr_of_mass , angle = np.pi/1.5).matrix @ rand_transform
            # new_pose = RigidTransform.rotation_from_axis_and_origin(axis = [0,1,0], origin=np.zeros(3), angle = np.pi/4).matrix

            x_mesh, y_mesh, z_mesh = obj_mesh.extents
            for i in range(num_slices):
                mesh_cpy = obj_mesh.copy()
                extrude_matrix = np.eye(4)
                extrude_matrix[2,3] = - z_mesh / 1.5 * i / num_slices
                mesh_cpy.apply_transform(extrude_matrix)
                print(mesh_cpy.vertices.min(0),mesh_cpy.vertices.max(0))
                slice_list.append(mesh_cpy)
            #TODO start with the object outside of the table, and take many more steps, like 100.
            #TODO step by only 1mm to avoid disappearing meshes
            k=0
            while True:
                slice_mesh = trimesh.boolean.difference(slice_list, 'blender')
                if slice_mesh.extents is not None and slice_mesh.extents[0] > 3.9 and slice_mesh.extents[1] > 3.9:
                    break
                k+=1
                print(k) if k % 100 == 0 else 0
            # center_trans = np.eye(4)
            # center_trans[:3,3] +=  ctr_of_mass
            # slice_mesh.apply_transform(center_trans)
            slice_mesh.export("plots/slice_mesh.obj")

            slice_obj_mesh = Mesh.from_trimesh(slice_mesh)
            slice_object_node = Node(mesh=slice_obj_mesh, matrix=np.eye(4))
            scene2.add_node(slice_object_node)
            # scene2.set_pose(slice_object_node, pose=np.eye(4))
            flags = RenderFlags.DEPTH_ONLY #| RenderFlags.SKIP_CULL_FACES | RenderFlags.ALL_WIREFRAME
            # flags2 = RenderFlags.SHADOWS_DIRECTIONAL
            # image2 = renderer.render(scene2, flags=flags)
            image2 = renderer.render(scene2)[0]
            scene2.remove_node(slice_object_node)

            # image2 = negative_depth(image2, ctr_of_mass)

            # image_1 = Cut_Image(image1)
            # image1, image2 = Zero_BG(image1, DR=False), Zero_BG(image2, DR=False)

            if config['debug']:
                Plot_Image(image1, "test.png")
                # image2 = image1.copy()
                # image2[image2 < 0.7999] += 2 * (0.8 - image2[image2 < 0.7999])
                print(np.min(image1), np.min(image1[image1 != 0]), np.max(image1))
                print(np.min(image2), np.min(image2[image2 != 0]), np.max(image2))
                Plot_Image(image2, "test2.png")
                Plot_Datapoint(image1, image2, random_quat)
                # dataset.flush()
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

            print("Added object", obj_id, "and overall datapoints are:", data_point_counter, 
                                "in", round(time.time() - start_time, 2), "seconds")
    objects_added = np.array(list(objects_added.keys()),dtype=int)
    np.random.shuffle(objects_added)
    print("Added", data_point_counter, "datapoints to dataset from ", len(objects_added), "objects")
    print("Obj ID to split on training and validation:")
    print(objects_added[:len(objects_added)//5])
    dataset.flush()
