import numpy as np
import argparse
import os

from autolab_core import YamlConfig, RigidTransform, TensorDataset
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer2D as vis2d
import itertools
from dexnet.envs import NoRemainingSamplesException 

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
    dataset = TensorDataset("/nfs/diskstation/projects/rbt_2/", tensor_config)
    datapoint = dataset.datapoint_template
    
    labels = np.arange(6)
    transform_strs = ["45 X", "135 X", "45 Y", "135 Y", "45 Z", "135 Z"]

    i = 0
    while True:
        print('Object Number: ', i)
        try:
            env.reset()
        except NoRemainingSamplesException:
            break
        
        obj = env.state.obj
        
        transforms = [RigidTransform.rotation_from_axis_and_origin([1, 0, 0], obj.center_of_mass, np.pi/4), 
         RigidTransform.rotation_from_axis_and_origin([1, 0, 0], obj.center_of_mass, 3*np.pi/4), 
         RigidTransform.rotation_from_axis_and_origin([0, 1, 0], obj.center_of_mass, np.pi/4), 
         RigidTransform.rotation_from_axis_and_origin([0, 1, 0], obj.center_of_mass, 3*np.pi/4),
         RigidTransform.rotation_from_axis_and_origin([0, 0, 1], obj.center_of_mass, np.pi/4), 
         RigidTransform.rotation_from_axis_and_origin([0, 0, 1], obj.center_of_mass, 3*np.pi/4)]
        
        datapoint["depth_image1"] = env.observation.data
        transformed_results = [t * obj.T_obj_world for t in transforms]
        
        for label, transformed_obj, transform_str in zip(labels, transformed_results, transform_strs):
#             print(transform_str)
            env.state.obj.T_obj_world = transformed_obj
#             vis2d.imshow(env.observation, auto_subplot=True)
#             vis2d.show()
            datapoint["depth_image2"] = env.observation.data
            datapoint["transform_id"] = label
            dataset.add(datapoint)
            
        i += 1
        if i % 20 == 0:
            dataset.flush()
            

        
