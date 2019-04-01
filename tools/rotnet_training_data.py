import numpy as np
import argparse
import os

from autolab_core import YamlConfig, RigidTransform
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer2D as vis2d

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
    config = YamlConfig(args.config_filename)
    env = GraspingEnv(config, config['vis'])
    while True:
        try:
            env.reset()
        except NoRemainingSamplesException:
            break

        # compute all stable poses
        obj_config = config['state_space']['object']
        stable_poses, _ = env.state.obj.mesh.compute_stable_poses(
            sigma=obj_config['stp_com_sigma'],
            n_samples=obj_config['stp_num_samples'],
            threshold=obj_config['stp_min_prob']
        )
        for pose in stable_poses:
            rot, trans = RigidTransform.rotation_and_translation_from_matrix(pose)
            env.state.obj.T_obj_world = RigidTransform(rot, trans, 'obj', 'world')
            
            vis2d.figure()
            print env.observation.shape
            vis2d.imshow(env.observation, auto_subplot=True)
            vis2d.show()
        
