from pyrender import (Scene, PerspectiveCamera, Mesh, 
                      Viewer, OffscreenRenderer, RenderFlags)    


from autolab_core import YamlConfig, RigidTransform, TensorDataset
import trimesh
from sd_maskrcnn.envs import CameraStateSpace
import os
import numpy as np
import matplotlib.pyplot as plt

def update_scene(scene, pose_matrix):
    print("SCENE GET NODES")
    print(scene.get_nodes(name='object'))
    print("DONE SCENE GET NODES")
    # update workspace
    next(iter(scene.get_nodes(name='object'))).matrix = pose_matrix
    
def normalize(z):
    return z / np.linalg.norm(z)
    
if __name__ == "__main__":
    
    config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/unsup_rbt_data_gen.yaml'
    )
    
    config = YamlConfig(config_filename)
    
    scene = Scene()
    renderer = OffscreenRenderer(1, 1)
    
    # update camera
    
    cam = CameraStateSpace(config['state_space']['camera']).sample()
    
    camera = PerspectiveCamera(cam.yfov, znear=0.05, zfar=3.0,
                                   aspectRatio=cam.aspect_ratio)
    
    renderer.viewport_width = cam.width
    renderer.viewport_height = cam.height
    
    pose_m = cam.pose.matrix.copy()
    pose_m[:,1:3] *= -1.0
    scene.add(camera, pose=pose_m, name=cam.frame)
    scene.main_camera_node = next(iter(scene.get_nodes(name=cam.frame)))
    
    mesh_dir = os.path.join(config['state_space']['heap']['objects']['mesh_dir'], 'thingiverse')
    obj_config = config['state_space']['heap']['objects']
    
    for mesh_filename in os.listdir(mesh_dir):
        mesh = trimesh.load_mesh(os.path.join(mesh_dir,mesh_filename))
        obj_mesh = Mesh.from_trimesh(mesh)
        scene.add(obj_mesh, pose=np.eye(4), name='object')
          
        stable_poses, _ = mesh.compute_stable_poses(
            sigma=obj_config['stp_com_sigma'],
            n_samples=obj_config['stp_num_samples'],
            threshold=obj_config['stp_min_prob']
        )
        
        for pose_matrix in stable_poses:
            for i in range(10):
                rand_axis = normalize(np.random.rand(3))
                random_rotation = RigidTransform.rotation_from_axis_and_origin(rand_axis, mesh.center_mass, np.random.rand())
                obj.T_obj_world = random_rotation * obj.T_obj_world
            
            
            
            print("POSE MATRIX")
            print(pose_matrix)
            update_scene(scene, pose_matrix)
            image = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
            plt.imshow(image, cmap='gray')
            plt.show()
            
            
        
        
        
        
        
        
    
