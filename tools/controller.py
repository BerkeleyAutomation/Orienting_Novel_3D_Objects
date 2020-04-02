import io
import os
import torch
import numpy as np
from unsupervised_rbt.models import ResNetSiameseNetwork, InceptionSiameseNetwork
from unsupervised_rbt.losses.shapematch import ShapeMatchLoss
from autolab_core import YamlConfig, RigidTransform
import torchvision
import cv2
from unsupervised_rbt import TensorDataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset.open(os.path.join('/nfs/diskstation/projects/unsupervised_rbt', "elephant60"))
    config = "cfg/tools/unsup_rbt_train_quat.yaml"
    config = YamlConfig(config)
    model = ResNetSiameseNetwork(config['pred_dim'], n_blocks=config['n_blocks'], embed_dim=config['embed_dim']).to(device)
    model.load_state_dict(torch.load("models/564obj_1024.pt"))
    model.eval()

    os.system("blender --python blender/controller_render.py --pair")

    goal_image = "blender/image_0000.png"
    cur_image = "blender/image_0001.png"

    I_s, I_g = cv2.imread(cur_image,-1) / 65535 * 255, cv2.imread(goal_image,-1) / 65535 * 255
    im1_batch = torch.Tensor(torch.from_numpy([I_s]).float()).to(device)
    im2_batch = torch.Tensor(torch.from_numpy([I_g]).float()).to(device)

    print(im1_batch.size())

    pred_q = model(im1_batch, im2_batch)
    pred = torch.to_numpy(pred_q).to_list()
    
    pickle.dump(pred, open("blender/cur_quat.p", "wb"))

    os.system("blender --python blender/controller_render.py --pair")

