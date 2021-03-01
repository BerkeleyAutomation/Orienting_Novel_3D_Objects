from autolab_core import YamlConfig, RigidTransform
from unsupervised_rbt import TensorDataset
import os

import numpy as np
import sys
import argparse

import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/data_gen_quat.yaml')
    parser.add_argument('-config', type=str, default=default_config_filename)
    parser.add_argument('-input_dataset', type=str, required=True)
    parser.add_argument('-output_dataset', type=str, required=True)
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    config = YamlConfig(args.config)
    # to adjust
    name_gen_dataset = args.input_dataset
    dataset_root_path = "/nfs/diskstation/projects/unsupervised_rbt/"
    
    # dataset configuration
    tensor_config = config['dataset']['tensors']
    dataset = TensorDataset(dataset_root_path + args.output_dataset, tensor_config)
    
    # load the old dataset
    old_dataset = TensorDataset.open(dataset_root_path + name_gen_dataset)
    
    # iterate over it and mix stuff up
    print("old dataset has datapoints: ", old_dataset.num_datapoints)
    idx = np.arange(0,old_dataset.num_datapoints, 512*64)
    print(idx)
    # np.random.shuffle(idx)
    
    # import pdb
    # pdb.set_trace()
    for i in tqdm(idx):
        batch = old_dataset.get_item_list(np.arange(i, np.min((old_dataset.num_datapoints,i+512*64))))

        depth_image1 = np.expand_dims(np.squeeze(batch["depth_image1"]),-1)
        depth_image2 = np.expand_dims(np.squeeze(batch["depth_image2"]),-1)
        obj_id = batch["obj_id"]
        lie = batch["lie"]
        quaternion = batch["quaternion"]
        pose_matrix = batch["pose_matrix"]
        # print(depth_image1.shape)

        idx_batch = np.arange(0,len(depth_image1))
        np.random.shuffle(idx_batch)
        for j in tqdm(idx_batch):
            datapoint = dataset.datapoint_template
            datapoint["depth_image1"] = depth_image1[j]
            datapoint["depth_image2"] = depth_image2[j]
            datapoint["obj_id"] = obj_id[j]
            datapoint["lie"] = lie[j]
            datapoint["quaternion"] = quaternion[j]
            datapoint["pose_matrix"] = pose_matrix[j]
            dataset.add(datapoint)
#         print("\n num datapoints in set: ", dataset.num_datapoints)
        
    dataset.flush()

