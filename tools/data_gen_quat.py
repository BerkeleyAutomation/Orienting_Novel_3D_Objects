'''
This script generates data for the self-supervised rotation prediction task
'''

from autolab_core import YamlConfig, RigidTransform, TensorDataset
from scipy.spatial.transform import Rotation
import os

# Use this if you are SSH
# os.environ["PYOPENGL_PLATFORM"] = 'egl'
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


def normalize(z):
    return z / np.linalg.norm(z)

def Generate_Quaternion():
    """Generate a random quaternion with conditions.
    To avoid double coverage and limit our rotation space, 
    we make sure the real component is positive and have 
    the greatest magnitude. We also limit rotations to less
    than 60 degrees. We sample according to the following links:
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
    http://planning.cs.uiuc.edu/node198.html
    """
    quat = np.zeros(4)
    while np.max(np.abs(quat)) < 0.866:
      uniforms = np.random.uniform(0, 1, 3)
      one_minus_u1, u1 = np.sqrt(1 - uniforms[0]), np.sqrt(uniforms[0])
      uniforms_pi = 2*np.pi*uniforms
      quat = np.array([one_minus_u1*np.sin(uniforms_pi[1]), one_minus_u1*np.cos(uniforms_pi[1]), u1*np.sin(uniforms_pi[2]), u1*np.cos(uniforms_pi[2])])

    max_i = np.argmax(np.abs(quat))
    quat[3], quat[max_i] = quat[max_i], quat[3]
    if quat[3] < 0:
        quat = -1 * quat
    return quat


def Generate_Quaternion_i():
    """Generate a random quaternion with conditions.
    To avoid double coverage and limit our rotation space, 
    we make sure the i component is positive and
    has the greatest magnitude.
    """
    quat = np.random.uniform(-1, 1, 4)
    quat = normalize(quat)
    max_i = np.argmax(np.abs(quat))
    quat[0], quat[max_i] = quat[max_i], quat[0]
    if quat[0] < 0:
        quat = -1 * quat
    return quat

def Quaternion_String(quat):
    """Converts a 4 element quaternion to a string for printing
    """
    quat = np.round(quat, 3)
    return str(quat[3]) + " + " + str(quat[0]) + "i + " + str(quat[1]) + "j + " + str(quat[2]) + "k"

def Quaternion_to_Rotation(quaternion, center_of_mass):
    """Take in an object's center of mass and a quaternion, and
    return a rotation matrix.
    """
    rotation_vector = Rotation.from_quat(quaternion).as_rotvec()
    angle = np.linalg.norm(rotation_vector)
    axis = rotation_vector / angle
    return RigidTransform.rotation_from_axis_and_origin(axis=axis, origin=center_of_mass, angle=angle).matrix

def Generate_Random_Transform(center_of_mass):
    """Create a matrix that will randomly rotate an object about the z-axis by a random angle.
    """
    z_angle = 2*np.pi*np.random.random()
    return RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=center_of_mass, angle=z_angle).matrix, z_angle

def Plot_Datapoint(datapoint):
    """Takes in a datapoint of our Tensor Dataset, and plots its two images for visualizing their 
    iniitial pose and rotation.
    """
    plt.figure(figsize=(14,7))
    plt.subplot(121)
    fig1 = plt.imshow(datapoint["depth_image1"][:, :, 0], cmap='gray')
    plt.title('Stable pose')
    plt.subplot(122)
    fig2 = plt.imshow(datapoint["depth_image2"][:, :, 0], cmap='gray')
    fig1.axes.get_xaxis().set_visible(False)
    fig1.axes.get_yaxis().set_visible(False)
    fig2.axes.get_xaxis().set_visible(False)
    fig2.axes.get_yaxis().set_visible(False)
    plt.title('After Rigid Transformation: ' + Quaternion_String(datapoint["quaternion"]))
    plt.show()
    # plt.savefig("pictures/allobj/obj" + str(datapoint['obj_id']) + ".png")
    # plt.close()

def addNoise(image, std=0.001):
    """Adds noise to image array.
    """
    noise = np.random.normal(0, std, image.shape)
    return image + noise

def create_scene():
    """Create scene for taking depth images.
    """
    scene = Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
    renderer = OffscreenRenderer(viewport_width=1, viewport_height=1)

    # initialize camera and renderer
    cam = CameraStateSpace(config['state_space']['camera']).sample()

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
    parser.add_argument('--objpred', action='store_true')
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

    if args.objpred:
        num_samples_per_obj = config['num_samples_per_obj_objpred']
    else:
        num_samples_per_obj = config['num_samples_per_obj']

    obj_id = 0
    data_point_counter = 0
    num_too_similar_75 = 0
    num_too_similar_6 = 0
    num_too_similar_5 = 0
    num_too_similar_4 = 0
    num_too_similar_3 = 0
    symmetries = [3,3,3,0,0,3,0.5,1.5,1,3,0,1,2,3,1,0,1,3,2,2,3,3,3,3,1,0,0,1.5,2,2,2,3,1,1,3,0,3,1,1,2,0.5,3,
    0,1.5,1,1,3,1,0,1.5,2.5,1,0,3,3,0,3,3,1.5,2,3,0,3,2,3,1.5,3,1,2,3,3,2,1,3,1,3,0.5,2,1,0.5,1.5,3,2,2,2,2,
    2,3,2,2,3,0,3,3,3,2,0,1,2,3,3,3,3,0,3,0,3,3,3,2,1.5,3,1,3,3,3,3,3,0,3,0,3,1,0,3,3,2,0,3,3,3,2,1,3,1.5,3,
    3,0,3,2,2,2,2,2,2,2,2,3,0.5,0,2,3,2,3,3,2,3,2,2,3,2,1.5,3,0.5,3,2,2,1.5,2,3,3,3,3,3,3,2,2,3,2,3,2,2,0.5,
    2,2,2,3,2,0,3,1.5,0,2,0,0,2,2,2,0,0,1,1.5,2,2,0]

    for mesh_dir, mesh_list in zip(mesh_dir_list, mesh_lists):
        for mesh_filename in mesh_list:
            obj_id += 1
            # if obj_id != 4:
            #     continue
            # if obj_id > 20:
            #     break
                # dataset.flush()
                # sys.exit(0)
            if obj_id > len(symmetries) or symmetries[obj_id-1] != 0:
                continue

            print(colored('------------- Object ID ' + str(obj_id) + ' -------------', 'red'))

            # load object mesh
            mesh = trimesh.load_mesh(os.path.join(mesh_dir, mesh_filename))
            obj_mesh = Mesh.from_trimesh(mesh)
            object_node = Node(mesh=obj_mesh, matrix=np.eye(4))
            scene.add_node(object_node)
            # scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=np.eye(4)) # for rgb? 

            # calculate stable poses
            stable_poses, _ = mesh.compute_stable_poses(
                sigma=obj_config['stp_com_sigma'],
                n_samples=obj_config['stp_num_samples'],
                threshold=obj_config['stp_min_prob']
            )

            if len(stable_poses) == 0:
                scene.remove_node(object_node)
                continue

            for _ in range(max(num_samples_per_obj // len(stable_poses), 1)):
                # iterate over all stable poses of the object
                for j, pose_matrix in enumerate(stable_poses):
                    # print("Stable Pose number:", j)
                    ctr_of_mass = pose_matrix[0:3, 3]

                    # Render image 1, which will be our original image with a random initial pose
                    rand_transform, z_angle = Generate_Random_Transform(ctr_of_mass) @ pose_matrix
                    scene.set_pose(object_node, pose=rand_transform)
                    image1 = 1 - renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
                    # image1, depth = renderer.render(scene, RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL)

                    # Render image 2, which will be image 1 rotated according to our specification
                    random_quat = Generate_Quaternion()
                    quat_str = Quaternion_String(random_quat)
                    # print("Quaternion: ", quat_str)
                    # print("Rotation Matrix: ", Rotation.from_quat(random_quat).as_dcm())
                    # print("Random Rotation Matrix: ", Generate_Random_Transform(ctr_of_mass))
                    # print("Image 1: ", rand_transform)
                    new_pose = Quaternion_to_Rotation(random_quat, ctr_of_mass) @ rand_transform

                    scene.set_pose(object_node, pose=new_pose)
                    image2 = 1 - renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
                    # image2, depth = renderer.render(scene, RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL)
                    # print(image1[:,:,0].shape)
                    # image1 = image1[:,:,0]*0.3 + image1[:,:,1]*0.59 * image1[:,:,2]*0.11
                    # image2 = image2[:,:,0]*0.3 + image2[:,:,1]*0.59 * image2[:,:,2]*0.11

                    mse = np.linalg.norm(image1 - image2)
                    # if mse < 0.75:
                    # if mse < 0.6:
                        # if config['debug']:
                        # print("Too similar MSE:", mse)
                        # print("Quaternion is ", 180/np.pi*np.linalg.norm(Rotation.from_quat(random_quat).as_rotvec()))
                        # num_too_similar += 1
                    # else:
                    #     if config['debug']:
                    #     print("MSE okay:", mse)
                    if mse < 0.75:
                        num_too_similar_75 += 1
                    if mse < 0.6:
                        num_too_similar_6 += 1
                    if mse < 0.5:
                        num_too_similar_5 += 1
                    if mse < 0.4:
                        num_too_similar_4 += 1
                    if mse < 0.3:
                        num_too_similar_3 += 1
                    image1, image2 = addNoise(image1, config['noise']), addNoise(image2, config['noise'])

                    datapoint = dataset.datapoint_template
                    datapoint["depth_image1"] = np.expand_dims(image1, -1)
                    datapoint["depth_image2"] = np.expand_dims(image2, -1)
                    datapoint["quaternion"] = random_quat
                    datapoint["obj_id"] = obj_id
                    datapoint["pose_matrix"] = pose_matrix
                    datapoint["z_angle"] = z_angle

                    # if num_too_similar < 2 or num_second_dp_match < 3 or True:
                    if mse >= 0.75:
                        # print("ADDING STABLE POSE ", j)
                        if config['debug']:
                            Plot_Datapoint(datapoint)

                        data_point_counter += 1
                        dataset.add(datapoint)
                    # else:
                    #     print("Not ADDING STABLE POSE")
            print("Added object ", obj_id, " and overall datapoints are: ", data_point_counter)
            # delete the object to make room for the next
            scene.remove_node(object_node)
    print("Added ", data_point_counter, " datapoints to dataset")
    print("Number of datapoints with difference 0.75,0.6,0.5,0.4,0.3 is", num_too_similar_75, 
            num_too_similar_6, num_too_similar_5, num_too_similar_4, num_too_similar_3)
    dataset.flush()
