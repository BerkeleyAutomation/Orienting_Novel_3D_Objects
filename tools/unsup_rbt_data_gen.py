from pyrender import (Scene, PerspectiveCamera, Mesh, 
                      Viewer, OffscreenRenderer, RenderFlags)    


from autolab_core import YamlConfig, RigidTransform, TensorDataset
import trimesh
from sd_maskrcnn.envs import CameraStateSpace
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from termcolor import colored

# for setting up offscreen rendering using egl
# os.environ["PYOPENGL_PLATFORM"] = 'osmesa'

def update_scene(scene, pose_matrix):
    # update workspace
    next(iter(scene.get_nodes(name='object'))).matrix = pose_matrix
    
def normalize(z):
    return z / np.linalg.norm(z)
    
if __name__ == "__main__":
    # to adjust
    name_gen_dataset = 'z-axis-only' 
    transform_strs = ["0 Z", "45 Z", "90 Z", "135 Z"]
    
    # TO DISCUSS: 200x200x1 pixel setting now (in the yaml file)
    # setup configurations from file
    config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/unsup_rbt_data_gen.yaml')
    config = YamlConfig(config_filename)
    scene = Scene()
    renderer = OffscreenRenderer(viewport_width=1, viewport_height=1)
    
    # dataset configuration
    tensor_config = config['dataset']['tensors']
    dataset = TensorDataset("/nfs/diskstation/projects/unsupervised_rbt/"+ name_gen_dataset + "/", tensor_config)
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
    
    # get a lost of all meshes
    mesh_dir = os.path.join(config['state_space']['heap']['objects']['mesh_dir'], 'thingiverse')
    obj_config = config['state_space']['heap']['objects']
    mesh_list = os.listdir(mesh_dir)
    
    i = 0
    data_point_counter = 0
    while True:
        # log
        print(colored('------------- Object Number ' + str(i) + ' -------------', 'red'))
        i += 1
        
        # get random item from the meshes
        obj_id = random.choice(range(len(mesh_list)))
        mesh_filename = mesh_list[obj_id]
        print('Object Name: ', mesh_filename)
        # delete object name in the list (that we don't sample it multiple times)
        del mesh_list[obj_id]
        
        # load object mesh
        mesh = trimesh.load_mesh(os.path.join(mesh_dir,mesh_filename))
        obj_mesh = Mesh.from_trimesh(mesh)
        scene.add(obj_mesh, pose=np.eye(4), name='object')
        
        # calculate stable poses
        stable_poses, _ = mesh.compute_stable_poses(
            sigma=obj_config['stp_com_sigma'],
            n_samples=obj_config['stp_num_samples'],
            threshold=obj_config['stp_min_prob']
        )
        
        # iterate over all stable poses of the object
        for j, pose_matrix in enumerate(stable_poses):
            print("Stable Pose number:", j)
#             print("pose matrix", pose_matrix)
            ctr_of_mass = pose_matrix[0:3,3]
#             print("center of mass", ctr_of_mass)
            
            # set up the transformations of which one is chosen at random per stable pose
            transforms = [
                RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=0), 
                RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=np.pi/2), 
                RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=np.pi), 
                RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=ctr_of_mass, angle=3*np.pi/2)
                ]

            if len(transform_strs) != len(transforms):
                print("Error: the number of elements in transform_strs and transforms are not equal")
                os._exit(1)

            # get image 1 which is the stable pose
            update_scene(scene, pose_matrix)
            image1 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
#             print("image 1 pose \n", pose_matrix)
            
            # pick a transform
            transform_id = np.random.choice(np.arange(len(transform_strs)))
            new_pose, tr_str = transforms[transform_id].matrix @ pose_matrix, transform_strs[transform_id]
#             print("image 2 pose \n", new_pose)
            
            # get image 2 which is the rotated image
            update_scene(scene, new_pose)
            image2 = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
            
#             plt.subplot(121)
#             plt.imshow(image1, cmap='gray')
#             plt.title('Stable pose')
#             plt.subplot(122)
#             plt.imshow(image2, cmap='gray')
#             plt.title('After Rigid Transformation: ' + tr_str)
#             plt.show()
            
            # safe as datapoint and add to dataset
            datapoint["depth_image1"] = image1
            datapoint["depth_image2"] = image2
            datapoint["transform_id"] = transform_id
            data_point_counter += 1
            dataset.add(datapoint)
            
            if data_point_counter == 100:
                dataset.flush()
                data_point_counter = 0
            
            
        
        
        
        
        
        
    
