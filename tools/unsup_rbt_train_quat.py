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

from tools.data_gen_quat import Quaternion_String, create_scene, Quaternion_to_Rotation

import trimesh
from pyrender import (Scene, IntrinsicsCamera, Mesh,
                      Viewer, OffscreenRenderer, RenderFlags, Node)


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

    # train_indices = dataset.split('train')[0][:10000]
    train_indices = dataset.split('train')[0]
    # train_indices = dataset.split('train2')[0]
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

    # test_indices = dataset.split('train')[1][:1000]
    test_indices = dataset.split('train')[1][:64*100]
    # test_indices = dataset.split('train2')[1][:64*100]
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

def Plot_Angle_vs_Loss(quaternions, losses, mean_loss, max_angle = 30):
    bins = max_angle // 5
    rotation_angles = [[] for i in range(bins)]
    for q,l in zip(quaternions, losses):
        rot_vec = Rotation.from_quat(q).as_rotvec()
        rot_angle = np.linalg.norm(rot_vec) * 180 / np.pi
        bin_num = np.min((int(rot_angle // 5), bins-1))
        rotation_angles[bin_num].append(l)

    mean_losses = [np.mean(ra) for ra in rotation_angles]
    errors = [np.std(ra) for ra in rotation_angles]
    labels = [str(i) + "-" + str(i+5) for i in range(0,max_angle,5)]

    plt.figure(figsize=(10,5))
    plt.bar(labels, mean_losses, yerr = errors)
    plt.axhline(mean_loss, c = 'r')
    plt.xlabel("Rotation Angle (Degrees)")
    plt.ylabel("Angle Loss (Degrees)")
    plt.ylim(0.0, (np.max(mean_losses)+np.max(errors))*1.1)
    plt.title("Loss vs Rotation Angle")
    plt.savefig(config['rotation_predictions_plot'])
    plt.close()

def Plot_Axis_vs_Loss(quaternions, losses, mean_loss):
    bins = 9
    rotation_angles = [[] for i in range(bins)]
    for q,l in zip(quaternions, losses):
        rot_vec = Rotation.from_quat(q).as_rotvec()
        theta_from_z = np.arccos(np.abs(rot_vec[2] / np.linalg.norm(rot_vec))) * 180 / np.pi
        bin_num = int(theta_from_z // 10)
        rotation_angles[bin_num].append(l)

    labels = [str(i) + "-" + str(i+10) for i in range(0,90,10)]
    mean_losses = [np.mean(ra) for ra in rotation_angles]
    errors = [np.std(ra) for ra in rotation_angles]
    plt.figure(figsize=(10,5))
    plt.bar(labels, mean_losses, yerr = errors)
    plt.axhline(mean_loss, c = 'r')
    plt.xlabel("Rotation Angle from Z-Axis (Degrees)")
    plt.ylabel("Angle Loss (Degrees)")
    plt.ylim(0.0, (np.max(mean_losses)+np.max(errors))*1.1)
    plt.title("Loss vs Rotation Angle from Z-Axis")
    plt.savefig("plots/axes_loss.png")
    plt.close()

def Plot_Bad_Predictions(dataset, predicted_quats, indices, name = "worst"):
    """Takes in the dataset, predicted quaternions, and indices of the 
    worst predictions in the validation set
    """
    for i in indices:
        datapoint = dataset.get_item_list(test_indices[i:i+1])
        predicted_quat = predicted_quats[i]
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        fig1 = plt.imshow(datapoint["depth_image1"][0][0], cmap='gray', vmin=np.min(datapoint["depth_image1"][0][0]))
        plt.title('Stable pose')
        plt.subplot(132)
        fig2 = plt.imshow(datapoint["depth_image2"][0][0], cmap='gray')
        plt.title('True Quat: ' + Quaternion_String(datapoint["quaternion"][0]))
        plt.subplot(133)
        fig3 = plt.imshow(Plot_Predicted_Rotation(datapoint, predicted_quat), cmap='gray')
        plt.title('Pred Quat: ' + Quaternion_String(predicted_quat))
        fig1.axes.get_xaxis().set_visible(False)
        fig1.axes.get_yaxis().set_visible(False)
        fig2.axes.get_xaxis().set_visible(False)
        fig2.axes.get_yaxis().set_visible(False)
        fig3.axes.get_xaxis().set_visible(False)
        fig3.axes.get_yaxis().set_visible(False)
        plt.savefig("plots/worst_preds/" + name + "_pred_" + str(datapoint['obj_id'][0]) + "_"
        + str(1- np.dot(predicted_quat, datapoint['quaternion'].flatten()))[2:5])
        print(1 - np.dot(predicted_quat, datapoint['quaternion'].flatten()))
        # plt.show()
        plt.close()

def Plot_Predicted_Rotation(datapoint, predicted_quat):
    object_id, pose_matrix = datapoint['obj_id'], datapoint['pose_matrix'][0]
    config = YamlConfig(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                                     'cfg/tools/data_gen_quat.yaml'))
    scene, renderer = create_scene(data_gen=False)
    dataset_name_list = ['3dnet', 'thingiverse', 'kit']
    mesh_dir = config['state_space']['heap']['objects']['mesh_dir']
    mesh_dir_list = [os.path.join(mesh_dir, dataset_name) for dataset_name in dataset_name_list]
    obj_config = config['state_space']['heap']['objects']
    mesh_lists = [os.listdir(mesh_dir) for mesh_dir in mesh_dir_list]
    obj_id = 0
    for mesh_dir, mesh_list in zip(mesh_dir_list, mesh_lists):
        for mesh_filename in mesh_list:
            obj_id += 1
            if obj_id != object_id:
                continue

            # load object mesh
            mesh = trimesh.load_mesh(os.path.join(mesh_dir, mesh_filename))
            obj_mesh = Mesh.from_trimesh(mesh)
            object_node = Node(mesh=obj_mesh, matrix=np.eye(4))
            scene.add_node(object_node)
            # print(pose_matrix)
            ctr_of_mass = pose_matrix[0:3, 3]

            new_pose = Quaternion_to_Rotation(predicted_quat, ctr_of_mass) @ pose_matrix
            scene.set_pose(object_node, pose=new_pose)
            return renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)



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
    parser.add_argument('--worst_pred', action='store_true')
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
        elephant_small_angle: small angles. 4000 datapoints
        elephant_noise: small angles, N(0,0.002) noise. 6000 datapoints
        800obj_200rot_v2: small angles, no noise, small differences removed (). Upto 200 rotations per OBJECT. ~90,000 datapoints
        no_symmetry_30obj_400rot: small angles, no noise, small diffs not removed, 400 per OBJECT, 11986 datapoints
        nosym_30obj_1000rot: above w 1000 rotations
        nosym_29obj_1000rot: above w small diffs 0.75 removed, no more shoe
        nosym_47obj_1000rot: no noise, 45 degrees w small diffs 0.5 removed. Added pose matrix. Train/Test split diff, has unseen in test
        72obj_2000rot: no noise, 30 degrees w small diffs 0.4 removed. Removed some more bad obj. 57 train, 15 test, 137679 points
        best_scores_2000rot: first attempt at score function
        72obj_random_rot: mse 0.5, no more z-axis, random 0-45. stable pose threshold down from 0.10 to 0.08, 136,098 points
        72obj_cuts: 135,440 points, image 1 random cuts
        72obj_occ: ~130,000 points, image 1 random rectangles
        uniform30, uniform45, uniform60: ~143,940 points, occlusions, angles sampled uniformly from 0-30,0-45,0-60
    """
    args = parse_args()
    config = YamlConfig(args.config)

    dataset = TensorDataset.open(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetSiameseNetwork(config['pred_dim'], n_blocks=config['n_blocks'], embed_dim=config['embed_dim']).to(device)
#         model = InceptionSiameseNetwork(config['pred_dim']).to(device)
    loss_func = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters())
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.8)
    if not args.test:
        if not os.path.exists(args.dataset + "/splits/train"):
            obj_id_split = np.loadtxt("cfg/tools/train_split")
            val_indices = []
            for i in range(dataset.num_datapoints):
                if dataset.datapoint(i)["obj_id"] in obj_id_split:
                    val_indices.append(i)

            print("Created Train Split")
            # dataset.make_split("train", train_pct=0.8)
            dataset.make_split("train", train_pct=0.8, val_indices= val_indices)
        if not os.path.exists(args.dataset + "/splits/train2"):
            dataset.make_split("train2", train_pct=0.8)

        train_losses, test_losses = [], []
        min_loss = 100000
        for epoch in range(config['num_epochs']):

            train_loss = train(dataset, config['batch_size'])
            test_loss = test(dataset, config['batch_size'])
            # scheduler.step()
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
        # model.load_state_dict(torch.load(config['final_epoch_dir']))
        model.load_state_dict(torch.load(config['best_epoch_dir']))
        # display_conv_layers(model)
        model.eval()
        test_loss, total = 0, 0

        # test_indices = dataset.split('train')[1][:1000]
        test_indices = dataset.split('train')[1]
        # test_indices = dataset.split('train2')[1]
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

                loss = loss_func(pred_transform, transform_batch, ones).item()
                true_quaternions.extend(transform_batch.cpu().numpy())
                pred_quaternions.extend(pred_transform.cpu().numpy())

                angle_loss = np.arccos(1-loss) * 180 / np.pi
                losses.append(angle_loss)
                test_loss += angle_loss
        print("Mean loss is: ", test_loss/total)
        Plot_Angle_vs_Loss(true_quaternions, losses, test_loss/total)
        Plot_Axis_vs_Loss(true_quaternions, losses, test_loss/total)
        if args.worst_pred:
            biggest_losses = np.argsort(losses)[-10:-1]
            smallest_losses = np.argsort(losses)[:10]
            Plot_Bad_Predictions(dataset, pred_quaternions, biggest_losses)
            Plot_Bad_Predictions(dataset, pred_quaternions, smallest_losses, "best")
        
        Plot_Loss(config)


