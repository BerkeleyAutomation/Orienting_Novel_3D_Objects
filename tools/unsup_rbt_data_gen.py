from autolab_core import YamlConfig, RigidTransform, TensorDataset
import os
os.environ["PYOPENGL_PLATFORM"] = 'osmesa'

import numpy as np
import trimesh
import itertools
import sys

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
    return scene, renderer

    
if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/unsup_rbt_data_gen.yaml')
    config = YamlConfig(config_filename)
    # to adjust
    name_gen_dataset = 'z-axis-only' 
    if config['debug']:
        name_gen_dataset += "_junk"
        
    if name_gen_dataset.startswith('z-axis-only'):
        transform_strs = ["0 Z", "90 Z", "180 Z", "270 Z"]
    elif name_gen_dataset.startswith('xyz-axis'):
        transform_strs = ["0 Z", "90 X", "90 Y", "90 Z"]
    else:
        assert(False)
    
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
    data_point_counter = 0
    
    for mesh_dir, mesh_list in zip(mesh_dir_list, mesh_lists):
        for mesh_filename in mesh_list:
            obj_id += 1
#             if obj_id == 50:
#                 dataset.flush()
#                 sys.exit(0)
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
            
            for _ in range(config['num_samples_per_obj']):             
                # iterate over all stable poses of the object
                for j, pose_matrix in enumerate(stable_poses):
                    print("Stable Pose number:", j)
                    ctr_of_mass = pose_matrix[0:3,3]

                    # set up the transformations of which one is chosen at random per stable pose
                    if name_gen_dataset.startswith('xyz-axis'):
                        transforms = [
                            RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=0), 
                            RigidTransform.rotation_from_axis_and_origin(axis=[1, 0, 0], origin=ctr_of_mass, angle=np.pi/2), 
                            RigidTransform.rotation_from_axis_and_origin(axis=[0, 1, 0], origin=ctr_of_mass, angle=np.pi/2), 
                            RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=np.pi/2)
                            ]
                    elif name_gen_dataset.startswith('z-axis-only'):
                        transforms = [
                            RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=0), 
                            RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=np.pi/2), 
                            RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=np.pi), 
                            RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=3*np.pi/2)
                            ]
                    else:
                        assert(False)

                    rand_transform = RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=2*np.pi*np.random.random())
                    pose_matrix = rand_transform.matrix @ pose_matrix
                    # get image 1 which is the stable pose
                    scene.set_pose(object_node, pose=pose_matrix)
                    image1 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)

                    # iterate over all transforms
                    obj_datapoints = []             
                    num_too_similar = 0

                    for transform_id in range(len(transform_strs)):       
                        new_pose, tr_str = transforms[transform_id].matrix @ pose_matrix, transform_strs[transform_id]
                        scene.set_pose(object_node, pose=new_pose)
                        #update_scene(scene, new_pose)
                        image2 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)

                        if config['debug']:
                            plt.subplot(121)
                            plt.imshow(image1, cmap='gray')
                            plt.title('Stable pose')
                            plt.subplot(122)
                            plt.imshow(image2, cmap='gray')
                            plt.title('After Rigid Transformation: ' + tr_str)
                            plt.show()
                            print(transform_id)

                        mse = np.linalg.norm(image1-image2)
                        if mse < 0.75:
                            if config['debug']:
                                print("Too similar MSE:", mse)
                            num_too_similar += 1
                        else:
                            if config['debug']:
                                print("MSE okay:", mse)

                        datapoint = dataset.datapoint_template
                        datapoint["depth_image1"] = np.expand_dims(image1, -1)
                        datapoint["depth_image2"] = np.expand_dims(image2, -1)
                        datapoint["transform_id"] = transform_id
                        datapoint["obj_id"] = obj_id
                        obj_datapoints.append(datapoint)

                    num_second_dp_match = 0
                    for dp1, dp2 in itertools.combinations(obj_datapoints, 2):
                        if np.linalg.norm(dp1['depth_image2'] - dp2['depth_image2']) < 0.75:
                            num_second_dp_match += 1
          
                    if num_too_similar < 2 or num_second_dp_match < 3:
                        print("ADDING STABLE POSE")
                        for dp in obj_datapoints:
                            dataset.add(dp)
                            data_point_counter += 1
                    else:
                        print("Not ADDING STABLE POSE")
                    
                    if data_point_counter % 1000 == 0:
                        dataset.flush()

            # delete the object to make room for the next
            scene.remove_node(object_node)
    dataset.flush()
