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
    dataset = TensorDataset.open(os.path.join('/nfs/diskstation/projects/unsupervised_rbt', "elephant60"))
    config = "cfg/tools/unsup_rbt_train_quat.yaml"
    config = YamlConfig(config)
    model = ResNetSiameseNetwork(4, n_blocks=1, embed_dim=1024, dropout=4).to(device)
    model.load_state_dict(torch.load("models/546objv5/cos_sm_blk1_emb1024_reg7_drop4.pt"))
    model.eval()

    data_path = "/nfs/diskstation/shivin/shivin_exp"
    objects = ['banana_3dnet','CatLying_800_tex', 'elephant_1_3dnet' , 'high_heel_3dnet', 'nerf_gun_base_3994236',  
    'piggy_2926476', 'tennis_shoe_3dnet', 'hex_vase_3276254', 'mechanical_part_262891', 'nozzle_5222583', 'pipe_fork_5049153', 'two_hand_wrench_3133822']
    objects_dir = [data_path + "/" + obj + "/" + obj for obj in objects]
    labels_dir = [obj_dir + ".txt" for obj_dir in objects_dir]
    labels = np.loadtxt(labels_dir[2])
    dir_imgs_banana = [objects_dir[0]+ "_"+ str(i).zfill(4)+ ".png" for i in range(20)]
    dir_imgs_elephant = [objects_dir[2]+ "_"+ str(i).zfill(4)+ ".png" for i in range(20)]
    imgs = np.array([[cv2.imread(img_dir, -1)] for img_dir in dir_imgs_elephant]).astype(int)
    # for img in imgs[0]:
    #     for i in range(128):
    #         for j in range(128):
    #             if img[i][j] < 0.8:
    #                 img[i][j] = 0.3
    blender_table, blender_min = imgs[0][0].max()/65535, imgs[0][0].min()/65535
    blender_max = (imgs[0][0][imgs[0][0] != imgs[0][0].max()]).max()/65535
    print("Blender table:",blender_table, "Elephant min:", blender_min,"Elephant max:", blender_max )
    print(imgs[0][0])
    elephant = dataset.get_item_list([1])
    elephant_img1 = (elephant['depth_image1'] * 65535).astype(int)
    elephant_img2 = (elephant['depth_image2'] * 65535).astype(int)
    im1_batch = torch.Tensor(torch.from_numpy(elephant_img1).float()).to(device)
    im2_batch = torch.Tensor(torch.from_numpy(elephant_img2).float()).to(device)
    pyrender_table, pyrender_min = elephant_img2[0].max()/65535, elephant_img2[0].min()/65535
    pyrender_max = (elephant_img2[0][elephant_img2[0] != elephant_img2[0].max()]).max()/65535
    print("Pyrender table:", pyrender_table, "Elephant min:", pyrender_min,"Elephant max:",pyrender_max)
    print("Prediction on First Image:" , model(im1_batch,im2_batch).detach().cpu().numpy())
    imgs_batch = torch.Tensor(torch.from_numpy(imgs).float()).to(device)
    print(imgs_batch.size())
    imgs1, imgs2 = imgs_batch[::2],imgs_batch[1::2]
    print("Predictions:")
    print(model(imgs1,imgs2))
    print("Ground Truth:")
    print(labels)

    plt.figure()
    plt.subplot(121)
    plt.imshow(imgs[4][0],cmap='gray')
    plt.subplot(122)
    plt.imshow(imgs[5][0],cmap='gray')
    plt.savefig("blender/blender.png")

    plt.figure()
    plt.subplot(121)
    plt.imshow(elephant_img1[0][0],cmap='gray')
    plt.subplot(122)
    plt.imshow(elephant_img2[0][0],cmap='gray')
    plt.savefig("blender/pyrender.png")
