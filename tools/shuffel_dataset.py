from autolab_core import YamlConfig, RigidTransform
from autolab_core import TensorDataset
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
    parser.add_argument('-input_dataset', type=str, required=True)
    parser.add_argument('-output_dataset', type=str, required=True)
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    config = YamlConfig(args.config)
    # to adjust
    name_gen_dataset = args.input_dataset
    
    # dataset configuration
    tensor_config = config['dataset']['tensors']
    dataset = TensorDataset("/raid/mariuswiggert/" + args.output_dataset, tensor_config)
    
    # load the old dataset
    old_dataset = TensorDataset.open("/raid/mariuswiggert/" + name_gen_dataset)
    
    # iterate over it and mix stuff up
    print("old dataset has datapoints: ", old_dataset.num_datapoints)
    idx = np.arange(old_dataset.num_datapoints)
    np.random.shuffle(idx)
    
    for i in tqdm(idx):
        old_datapoint = old_dataset[i]
        
        datapoint = dataset.datapoint_template
        datapoint["depth_image1"] = old_datapoint["depth_image1"]
        datapoint["depth_image2"] = old_datapoint["depth_image2"]
        datapoint["transform_id"] = old_datapoint["transform_id"]
        datapoint["obj_id"] = old_datapoint["obj_id"]
        dataset.add(datapoint)
#         print("\n num datapoints in set: ", dataset.num_datapoints)
        
    dataset.flush()

