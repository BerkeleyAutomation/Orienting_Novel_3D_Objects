from autolab_core import YamlConfig, RigidTransform, TensorDataset
import os
os.environ["PYOPENGL_PLATFORM"] = 'osmesa'

import numpy as np
import trimesh

from pyrender import (Scene, PerspectiveCamera, Mesh, 
                      Viewer, OffscreenRenderer, RenderFlags)   

from sd_maskrcnn.envs import CameraStateSpace

import matplotlib.pyplot as plt
import random
from termcolor import colored

# for setting up offscreen rendering using egl
# assert(False)
# os.environ["PYOPENGL_PLATFORM"] = 'egl'

def update_scene(scene, pose_matrix):
    # update workspace
    next(iter(scene.get_nodes(name='object'))).matrix = pose_matrix
    
def normalize(z):
    return z / np.linalg.norm(z)
    
if __name__ == "__main__":
    # to adjust
    name_gen_dataset = 'z-axis-only' 
    transform_strs = ["0 Z", "90 X", "90 Y", "90 Z"]
    
    # TO DISCUSS: 200x200 pixel setting now (in the yaml file)
    # setup configurations from file
    config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/unsup_rbt_data_gen.yaml')
    config = YamlConfig(config_filename)
    scene = Scene()
    renderer = OffscreenRenderer(viewport_width=1, viewport_height=1)
    
    # dataset configuration
    tensor_config = config['dataset']['tensors']
    dataset = TensorDataset("/nfs/diskstation/projects/unsupervised_rbt/" + name_gen_dataset + "/", tensor_config)
    datapoint = dataset.datapoint_template
    
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
    
    # get a list of all meshes
    dataset_name_list = ['3dnet', 'thingiverse', 'kit']
    mesh_dir_list = [os.path.join(config['state_space']['heap']['objects']['mesh_dir'], dataset_name) for dataset_name in dataset_name_list]
    obj_config = config['state_space']['heap']['objects']
    mesh_list = [os.listdir(mesh_dir) for mesh_dir in mesh_dir_list]

    i = 0
    mesh_dir_idx = 0
    obj_idx = 0
    mesh_dir_lengths = [len(m) for m in mesh_list]
    print(mesh_dir_lengths)

    data_point_counter = 0
    while True:
        # log
        print(colored('------------- Object Number ' + str(i) + ' -------------', 'red'))
        i += 1
        
        # get random item from the meshes
        mesh_filename = mesh_list[mesh_dir_idx][obj_idx]
        obj_idx += 1
        
        print('Object Name: ', mesh_filename)
        
        # load object mesh
        mesh = trimesh.load_mesh(os.path.join(mesh_dir_list[mesh_dir_idx], mesh_filename))
        obj_mesh = Mesh.from_trimesh(mesh)
        scene.add(obj_mesh, pose=np.eye(4), name='object')
        
        # calculate stable poses
        stable_poses, _ = mesh.compute_stable_poses(
            sigma=obj_config['stp_com_sigma'],
            n_samples=obj_config['stp_num_samples'],
            threshold=obj_config['stp_min_prob']
        )
        
        if obj_idx == mesh_dir_lengths[mesh_dir_idx]:
            mesh_dir_idx += 1
            
        if mesh_dir_idx == len(mesh_dir_lengths):
            break
        
        # iterate over all stable poses of the object
        for j, pose_matrix in enumerate(stable_poses):
            print("Stable Pose number:", j)
#             print("pose matrix", pose_matrix)
            ctr_of_mass = pose_matrix[0:3,3]
#             print("center of mass", ctr_of_mass)
            
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

#             if len(transform_strs) != len(transforms):
#                 print("Error: the number of elements in transform_strs and transforms are not equal")
#                 os._exit(1)

            # get image 1 which is the stable pose
            update_scene(scene, pose_matrix)
            image1 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
#             print("image 1 pose \n", pose_matrix)
            
            # pick a transform
            for transform_id in range(len(transform_strs)):
                new_pose, tr_str = transforms[transform_id].matrix @ pose_matrix, transform_strs[transform_id]
                update_scene(scene, new_pose)
                image2 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)

#                 plt.subplot(121)
#                 plt.imshow(image1, cmap='gray')
#                 plt.title('Stable pose')
#                 plt.subplot(122)
#                 plt.imshow(image2, cmap='gray')
#                 plt.title('After Rigid Transformation: ' + tr_str)
#                 plt.show()
 
                datapoint["depth_image1"] = np.expand_dims(image1, -1)
                datapoint["depth_image2"] = np.expand_dims(image2, -1)
                datapoint["transform_id"] = transform_id
                data_point_counter += 1
                dataset.add(datapoint)
                
                if data_point_counter % 100 == 0:
                    dataset.flush()

            # transform_id = np.random.choice(np.arange(len(transform_strs)))
            # new_pose, tr_str = transforms[transform_id].matrix @ pose_matrix, transform_strs[transform_id]
            # print("image 2 pose \n", new_pose)
            
            # plt.subplot(121)
            # plt.imshow(image1, cmap='gray')
            # plt.title('Stable pose')
            # plt.subplot(122)
            # plt.imshow(image2, cmap='gray')
            # plt.title('After Rigid Transformation: ' + tr_str)
            # plt.show()
            
            # safe as datapoint and add to dataset
<<<<<<< HEAD
            # datapoint["depth_image1"] = np.expand_dims(image1, -1)
            # datapoint["depth_image2"] = np.expand_dims(image2, -1)
            # datapoint["transform_id"] = transform_id
            # data_point_counter += 1
            # dataset.add(datapoint)
=======
            datapoint["depth_image1"] = np.expand_dims(image1,-1)
            datapoint["depth_image2"] = np.expand_dims(image2,-1)
            datapoint["transform_id"] = transform_id
            data_point_counter += 1
            dataset.add(datapoint)
>>>>>>> 074d2b8539070ccce028afee108a5834eabe2f42
            
#             if data_point_counter == 100:
#                 dataset.flush()
#                 data_point_counter = 0
            
            
        
        
        
        
        
        
    
