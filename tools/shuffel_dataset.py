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
                                           'cfg/tools/data_gen.yaml')
    parser.add_argument('-config', type=str, default=default_config_filename)
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('--objpred', action='store_true')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    config = YamlConfig(args.config)
    # to adjust
    name_gen_dataset = args.dataset
    
    # dataset configuration
    tensor_config = config['dataset']['tensors']
    dataset = TensorDataset("/raid/mariuswiggert/" + name_gen_dataset + "_shuffled/", tensor_config)
    
    # load the old dataset
    old_dataset = TensorDataset.open("/raid/mariuswiggert/" + name_gen_dataset)
    
    # iterate over it and mix stuff up
    print("old dataset has datapoints: ", old_dataset.num_datapoints)
    idx = np.arange(old_dataset.num_datapoints)
    np.random.shuffle(idx)
    
    batch_size = 100
    
    for step in tqdm(range(old_dataset.num_datapoints//batch_size)):
        batch = old_dataset.get_item_list(idx[step*batch_size : (step+1)*batch_size])
        
        for i in range(batch_size):
            datapoint = dataset.datapoint_template
            datapoint["depth_image1"] = batch["depth_image1"][i].reshape((128, 128, 1))
            datapoint["depth_image2"] = batch["depth_image2"][i].reshape((128, 128, 1))
            datapoint["transform_id"] = batch["transform"][i]
            datapoint["obj_id"] = batch["obj_id"][i]
            dataset.add(datapoint)
            print("\n num datapoints in set: ", dataset.num_datapoints)

