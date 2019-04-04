import numpy as np
import argparse
import os

from autolab_core import YamlConfig, RigidTransform, TensorDataset
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer2D as vis2d
import itertools

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
    dataset = TensorDataset("/nfs/diskstation/projects/rigid_body/", tensor_config)
    datapoint = dataset.datapoint_template

    i = 0
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
        
        print("Object Number: " + str(i))
        for pose_pair in list(itertools.combinations(stable_poses, 2)):
            pose1, pose2 = pose_pair
            
            rot_pos1, trans_pos1 = RigidTransform.rotation_and_translation_from_matrix(pose1)
            pose1_transform = RigidTransform(rot_pos1, trans_pos1, 'obj', 'world')
            env.state.obj.T_obj_world = pose1_transform
            datapoint["depth_image1"] = env.observation.data
            
            rot_pos2, trans_pos2 = RigidTransform.rotation_and_translation_from_matrix(pose2)
            pose2_transform = RigidTransform(rot_pos2, trans_pos2, 'obj', 'world')
            env.state.obj.T_obj_world = pose2_transform
            datapoint["depth_image2"] = env.observation.data
            
            datapoint["transform"] = (pose1_transform.inverse() * pose2_transform).matrix
            dataset.add(datapoint)
            
        i += 1
        if i % 20 == 0:
            dataset.flush()
            

        
