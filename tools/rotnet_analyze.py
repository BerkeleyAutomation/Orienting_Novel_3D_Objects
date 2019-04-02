import numpy as np
import argparse
import os

from autolab_core import YamlConfig, RigidTransform, TensorDataset
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer2D as vis2d
import itertools
from perception import DepthImage, RgbdImage

def parse_args():
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/rotnet_training_data.yaml'
    )
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('--num_objs', type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
#     config = YamlConfig(args.config_filename)
#     env = GraspingEnv(config, config['vis'])
#     tensor_config = config['dataset']['tensors']
#     dataset = TensorDataset("rbt_dataset/", tensor_config)
#     datapoint = dataset.datapoint_template
    
    dataset = TensorDataset.open('rbt_dataset/')
    
    for datapoint in dataset:
        depth_image1 = RgbdImage(datapoint["depth_image1"])
        depth_image2 = RgbdImage(datapoint["depth_image2"])
        vis2d.figure()
        vis2d.imshow(depth_image1, auto_subplot=True)
        vis2d.show()
        
        vis2d.figure()
        vis2d.imshow(depth_image2, auto_subplot=True)
        vis2d.show()
        

        
