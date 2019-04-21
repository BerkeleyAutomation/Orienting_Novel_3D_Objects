import numpy as np
import argparse
import os

from autolab_core import YamlConfig, RigidTransform, TensorDataset
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer2D as vis2d
from dexnet.visualization import DexNetVisualizer3D as vis3d
import itertools
from dexnet.envs import NoRemainingSamplesException 

def normalize(z):
    return z / np.linalg.norm(z)

def parse_args():
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/rb_net_data_gen.yaml'
    )
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('--num_objs', type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = YamlConfig(args.config_filename)
    env = GraspingEnv(config, config['vis'])
    tensor_config = config['dataset']['tensors']
    dataset = TensorDataset("/nfs/diskstation/projects/unsupervised_rbt/axis_pred_2/", tensor_config)
    datapoint = dataset.datapoint_template
    
    labels = np.arange(4)
    transform_strs = ["45 X", "45 Y", "45 Z", "0"]

    i = 0
    while True:
        print('Object Number: ', i)
        try:
            env.reset()
        except NoRemainingSamplesException:
            break
        
        obj = env.state.obj

        rand_axis = normalize(np.random.rand(3))
        random_rotation = RigidTransform.rotation_from_axis_and_origin(rand_axis, obj.center_of_mass, np.random.rand())
        obj.T_obj_world = random_rotation * obj.T_obj_world

        #env.render_3d_scene()
        #vis3d.show()
        #vis2d.imshow(env.observation, auto_subplot=True)
        #vis2d.show()
        
        transforms = [
            RigidTransform.rotation_from_axis_and_origin([1, 0, 0], obj.center_of_mass, np.pi/2), 
#             RigidTransform.rotation_from_axis_and_origin([1, 0, 0], obj.center_of_mass, 3*np.pi/4), 
            RigidTransform.rotation_from_axis_and_origin([0, 1, 0], obj.center_of_mass, np.pi/2), 
#             RigidTransform.rotation_from_axis_and_origin([0, 1, 0], obj.center_of_mass, 3*np.pi/4),
            RigidTransform.rotation_from_axis_and_origin([0, 0, 1], obj.center_of_mass, np.pi/2), 
#             RigidTransform.rotation_from_axis_and_origin([0, 0, 1], obj.center_of_mass, 3*np.pi/4)
            RigidTransform.rotation_from_axis_and_origin([0, 0, 1], obj.center_of_mass, 0)
        ]
        
        label = np.random.choice(np.arange(4))
        new_tf, new_tf_str = transforms[label] * obj.T_obj_world, transform_strs[label]
        # print(new_tf_str)
        datapoint["depth_image1"] = env.observation.data
        env.state.obj.T_obj_world = new_tf
        #env.render_3d_scene()
        #vis3d.show()
        #vis2d.imshow(env.observation, auto_subplot=True)
        #vis2d.show()
        datapoint["depth_image2"] = env.observation.data
        datapoint["transform_id"] = label
        dataset.add(datapoint)
        # print datapoint
            
        i += 1
        if i % 20 == 0:
            dataset.flush()
            

        
