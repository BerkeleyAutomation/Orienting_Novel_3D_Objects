import io
import os
import torch
import numpy as np
from unsupervised_rbt.models import ResNetSiameseNetwork, InceptionSiameseNetwork
from unsupervised_rbt.losses.shapematch import ShapeMatchLoss
from autolab_core import YamlConfig, RigidTransform
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import cv2
from unsupervised_rbt import TensorDataset
from torch.autograd import Variable
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset.open(os.path.join('/nfs/diskstation/projects/unsupervised_rbt', "872obj"))
    config = "cfg/tools/unsup_rbt_train_quat.yaml"
    config = YamlConfig(config)
    model = ResNetSiameseNetwork(4, n_blocks=1, embed_dim=1024, dropout=4).to(device)
    model.load_state_dict(torch.load("models/872obj/cos_sm_blk1_emb1024_reg9_drop4.pt"))
    model.eval()


    for i in range(1, 4000, 200):
        pyrender = dataset.get_item_list([i])
        pyrender_img1 = pyrender['depth_image1']
        pyrender_img2 = pyrender['depth_image2']
        im1_batch = torch.Tensor(torch.from_numpy(pyrender_img1).float()).to(device)
        im2_batch = torch.Tensor(torch.from_numpy(pyrender_img2).float()).to(device)
        pyrender_max, pyrender_min = pyrender_img2[0].max(), pyrender_img2[0].min()
        pyrender_min = (pyrender_img2[0][pyrender_img2[0] != pyrender_img2[0].min()]).min()
        print("Pyrender max:", pyrender_max, "pyrender min:", pyrender_min)
        # print("Prediction on First Image:" , model(im1_batch,im2_batch).detach().cpu().numpy())
    # imgs_batch = torch.Tensor(torch.from_numpy(imgs).float()).to(device)
    # print(imgs_batch.size())
    # imgs1, imgs2 = imgs_batch[::2],imgs_batch[1::2]
    # print("Predictions:")
    # print(model(imgs1,imgs2))
    # print("Ground Truth:")
    # print(labels)

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(imgs[4][0],cmap='gray')
    # plt.subplot(122)
    # plt.imshow(imgs[5][0],cmap='gray')
    # plt.savefig("blender/blender.png")

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(elephant_img1[0][0],cmap='gray')
    # plt.subplot(122)
    # plt.imshow(elephant_img2[0][0],cmap='gray')
    # plt.savefig("blender/pyrender.png")
