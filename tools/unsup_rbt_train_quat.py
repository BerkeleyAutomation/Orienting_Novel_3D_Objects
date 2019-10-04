'''
Train self-supervised task (rotation prediction) task; current good dataset to use is quaternion_shivin or quaternion_objpred; 
Currently have a pre-trained model for this, which is referenced in semi_sup script
'''

import numpy as np
import argparse
import os
import itertools
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from autolab_core import YamlConfig, RigidTransform
from unsupervised_rbt import TensorDataset
from unsupervised_rbt.models import ResNetSiameseNetwork, InceptionSiameseNetwork
from perception import DepthImage, RgbdImage

from scipy.spatial.transform import Rotation

from tools.data_gen_quat import Quaternion_String

# class CosineLoss(nn.Module):
#     def __init__(self):
#         super(CosineLoss,self).__init__()
        
#     def forward(self,x,y):
#         print("X is: ", x, "Y is: ", y)
#         return 1 - x.dot(y)


def train(dataset, batch_size):
    '''Train model specified in main and return training loss and classification accuracy'''
    model.train()
    train_loss, total = 0, 0

    train_indices = dataset.split('train')[0][:10000]
    N_train = len(train_indices)
    n_train_steps = N_train//batch_size
    
    ones = torch.Tensor(np.ones(batch_size)).to(device)

    for step in tqdm(range(n_train_steps)):
        batch = dataset.get_item_list(train_indices[step*batch_size: (step+1)*batch_size])
        depth_image1 = (batch["depth_image1"] * 255).astype(int)
        depth_image2 = (batch["depth_image2"] * 255).astype(int)

        im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
        im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
        transform_batch = Variable(torch.from_numpy(batch["quaternion"])).to(device)

#         print(depth_image1.shape)
#         print(depth_image2.shape)

#         if step > 20:
#             for i in range(batch_size):
#                 plt.subplot(121)
#                 depth_image_show1 = depth_image1[i][0]
#                 plt.imshow(depth_image_show1, cmap='gray')
#                 plt.subplot(122)
#                 depth_image_show2 = depth_image2[i][0]
#                 plt.imshow(depth_image_show2, cmap='gray')
#                 plt.title('Transform: {}'.format(transform_batch[i]))
#                 plt.show()

        optimizer.zero_grad()
        pred_transform = model(im1_batch, im2_batch)
#         print("TRANSFORM BATCH")
#         print(transform_batch)
        # _, predicted = torch.max(pred_transform, 1)
#         print("PRED TRANSFORM")
#         print(predicted)
        # correct += (predicted == transform_batch).sum().item()
        total += transform_batch.size(0)
        # print(transform_batch)
        loss = loss_func(pred_transform, transform_batch, ones)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # class_acc = 100 * correct/total
    return train_loss/n_train_steps


def test(dataset, batch_size):
    """
    Return loss and classification accuracy of the model on the test data
    """
    model.eval()
    test_loss, total = 0, 0

    test_indices = dataset.split('train')[1][:1000]
    n_test = len(test_indices)
    n_test_steps = n_test // batch_size

    ones = torch.Tensor(np.ones(batch_size)).to(device)

    with torch.no_grad():
        for step in tqdm(range(n_test_steps)):
            batch = dataset.get_item_list(test_indices[step*batch_size: (step+1)*batch_size])
            depth_image1 = (batch["depth_image1"] * 255).astype(int)
            depth_image2 = (batch["depth_image2"] * 255).astype(int)
            im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
            im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
            transform_batch = Variable(torch.from_numpy(batch["quaternion"])).to(device)
            pred_transform = model(im1_batch, im2_batch)
#             print("TRUE TRANSFORMS")
#             print(transform_batch)
            # _, predicted = torch.max(pred_transform, 1)
#             print("PREDICTED TRANSFORMS")
#             print(predicted)
            # correct += (pred_transform == transform_batch).sum().item()
            total += transform_batch.size(0)

            loss = loss_func(pred_transform, transform_batch, ones)
            test_loss += loss.item()

    # class_acc = 100 * correct/total
    return test_loss/n_test_steps


def Plot_Loss(config):
    """Plots the training and validation loss, provided that there is a config file with correct
    location of data
    """
    losses = pickle.load(open(config['losses_f_name'], "rb"))
    train_returns = np.array(losses["train_loss"])
    test_returns = np.array(losses["test_loss"])

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(train_returns)) + 1, train_returns, label="Training Loss")
    plt.plot(np.arange(len(test_returns)) + 1, test_returns, label="Testing Loss")
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend(loc='best')
    plt.savefig(config['loss_plot_f_name'])
    plt.close()

def Plot_Angle_vs_Loss(quaternions, losses):
    rotation_angles = []
    for q in quaternions:
        rot_vec = Rotation.from_quat(q).as_rotvec()
        rotation_angles.append(np.linalg.norm(rot_vec))

    plt.figure(figsize=(10,5))
    plt.scatter(rotation_angles, losses)
    plt.xlabel("Rotation Angle")
    plt.ylabel("Loss")
    plt.ylim(0, 0.02)
    plt.title("Loss vs Rotation Angle")
    plt.savefig(config['rotation_predictions_plot'])
    plt.close()

def Plot_Bad_Predictions(dataset, predicted_quats, indices):
    """Takes in the dataset, predicted quaternions, and indices of the 
    worst predictions in the validation set
    """
    for i in indices:
        datapoint = dataset.get_item_list(test_indices[i:i+1])
        plt.figure(figsize=(14,7))
        plt.subplot(121)
        fig1 = plt.imshow(datapoint["depth_image1"][0][0], cmap='gray')
        plt.title('Stable pose')
        plt.subplot(122)
        fig2 = plt.imshow(datapoint["depth_image2"][0][0], cmap='gray')
        fig1.axes.get_xaxis().set_visible(False)
        fig1.axes.get_yaxis().set_visible(False)
        fig2.axes.get_xaxis().set_visible(False)
        fig2.axes.get_yaxis().set_visible(False)
        plt.title('True Quaternion: ' + Quaternion_String(datapoint["quaternion"][0]) + 
                '\n Predicted Quaternion: ' + Quaternion_String(predicted_quats[i]))
        plt.show()

def display_conv_layers(model):
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    with torch.no_grad():
        imshow(torchvision.utils.make_grid(model.resnet.conv1.weight))


def parse_args():
    """Parse arguments from the command line.
    -config to input your own yaml config file. Default is unsup_rbt_train_quat.yaml
    -dataset to input a name for your dataset. Should start with quaternion
    --test to generate a graph of your train and validation loss
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/unsup_rbt_train_quat.yaml')
    parser.add_argument('-config', type=str, default=default_config_filename)
    parser.add_argument('-dataset', type=str, required=True)
    args = parser.parse_args()
    args.dataset = os.path.join('/nfs/diskstation/projects/unsupervised_rbt', args.dataset)
    return args


if __name__ == '__main__':
    """Train on a dataset or generate a graph of the training and validation loss.
    Current Datasets: 
        quaternion_elephant: 2000 rotations per stable pose of object 4, an elephant. 8000 datapoints
        quaternion_800obj_200rot: 100 rotations per stable pose of 872 objects. 800*25*4*stable pose per obj = 175360 datapoints
        elephant_small_angle: smaller angles. 4000 datapoints
        elephant_noise: smaller angles, N(0,0.002) noise. 6000 datapoints

    """
    args = parse_args()
    config = YamlConfig(args.config)

    dataset = TensorDataset.open(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetSiameseNetwork(config['pred_dim'], n_blocks=1, embed_dim=20).to(device)
#         model = InceptionSiameseNetwork(config['pred_dim']).to(device)
    loss_func = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters())

    if not args.test:
        if not os.path.exists(args.dataset + "/splits/train"):
            print("Created Train Split")
            dataset.make_split("train", train_pct=0.8)

        train_losses, test_losses = [], []
        min_loss = 100000
        for epoch in range(config['num_epochs']):
            train_loss = train(dataset, config['batch_size'])
            test_loss = test(dataset, config['batch_size'])
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print("Epoch %d, Train Loss = %f, Test Loss = %f" %
                  (epoch, train_loss, test_loss))
            pickle.dump({"train_loss": train_losses, "test_loss": test_losses,
                        }, open(config['losses_f_name'], "wb"))
            torch.save(model.state_dict(), config['final_epoch_dir'])
            if test_loss < min_loss:
                torch.save(model.state_dict(), config['best_epoch_dir'])
                min_loss = test_loss

    else:
        model.load_state_dict(torch.load(config['final_epoch_dir']))
        # display_conv_layers(model)
        model.eval()
        test_loss, total = 0, 0

        test_indices = dataset.split('train')[1][:1000]
        n_test = len(test_indices)
        batch_size = 1
        ones = torch.Tensor(np.ones(batch_size)).to(device)
        n_test_steps = n_test // batch_size

        true_quaternions = []
        pred_quaternions = []
        losses = []
        with torch.no_grad():
            for step in tqdm(range(n_test_steps)):
                batch = dataset.get_item_list(test_indices[step*batch_size: (step+1)*batch_size])
                depth_image1 = (batch["depth_image1"] * 255).astype(int)
                depth_image2 = (batch["depth_image2"] * 255).astype(int)
                im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
                im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
                transform_batch = Variable(torch.from_numpy(batch["quaternion"])).to(device)
                pred_transform = model(im1_batch, im2_batch)
                # print("True Quaternions:")
                # print(transform_batch)
                # print("Predicted Quaternions:")
                # print(pred_transform)
                # correct += (pred_transform == transform_batch).sum().item()
                total += transform_batch.size(0)

                loss = loss_func(pred_transform, transform_batch, ones)
                true_quaternions.extend(transform_batch.cpu().numpy())
                pred_quaternions.extend(pred_transform.cpu().numpy())

                losses.append(loss.item())
                test_loss += loss.item()
        Plot_Angle_vs_Loss(true_quaternions, losses)
        biggest_losses = np.argsort(losses)[-10:-1]
        # smallest_losses = np.argsort(losses)[:10]
        # Plot_Bad_Predictions(dataset, pred_quaternions, biggest_losses)
        
        Plot_Loss(config)


