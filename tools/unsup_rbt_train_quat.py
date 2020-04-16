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
from unsupervised_rbt.losses.shapematch import ShapeMatchLoss
from perception import DepthImage, RgbdImage

from tools.data_gen_quat import create_scene
from tools.utils import *

import trimesh
from pyrender import (Scene, IntrinsicsCamera, Mesh,
                      Viewer, OffscreenRenderer, RenderFlags, Node)

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

def get_points(obj_ids, points_poses):
    points = [point_clouds[obj_id] / scales[obj_id] * 10 for obj_id in obj_ids]
    # print(batch["pose_matrix"][0])
    points = [points_poses[i] @ points[i] for i in range(len(obj_ids))]
    points = torch.Tensor(points).to(device)
    # print(points[:,:5])
    return points

def train(dataset, batch_size, first=False):
    '''Train model specified in main and return training loss and classification accuracy'''
    model.train()
    train_loss = 0

    # train_indices = dataset.split('train')[0][:10000]
    train_indices = dataset.split('train')[0]
    # train_indices = dataset.split('train2')[0][:10000]

    N_train = len(train_indices)
    n_train_steps = N_train//batch_size

    ones = torch.Tensor(np.ones(batch_size)).to(device)
    optimizer.zero_grad()

    for step in tqdm(range(n_train_steps)):
        batch = dataset.get_item_list(train_indices[step*batch_size: (step+1)*batch_size])
        # depth_image1 = Quantize(batch["depth_image1"])
        # depth_image2 = Quantize(batch["depth_image2"])
        depth_image1 = batch["depth_image1"]
        depth_image2 = batch["depth_image2"]

        im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
        im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
        transform_batch = Variable(torch.from_numpy(batch["quaternion"])).to(device)
        # if step > 20:
        #     for i in range(batch_size):
        #         plt.subplot(121)
        #         depth_image_show1 = depth_image1[i][0]
        #         plt.imshow(depth_image_show1, cmap='gray')
        #         plt.subplot(122)
        #         depth_image_show2 = depth_image2[i][0]
        #         plt.imshow(depth_image_show2, cmap='gray')
        #         plt.title('Transform: {}'.format(transform_batch[i]))
        #         plt.show()
        obj_ids = batch["obj_id"]
        points_poses = batch["pose_matrix"][:,:3,:3]
        points = get_points(obj_ids, points_poses)

        pred_transform = model(im1_batch, im2_batch)
        if config['loss'] == 'cosine' or first:
            loss = loss_func(pred_transform, transform_batch, ones)
            sm_loss = loss_func2(pred_transform, transform_batch, points).item()
        else:
            loss = loss_func2(pred_transform, transform_batch, points)
            sm_loss = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += sm_loss

        # if step % 100 == 0:
        #     print(pred_transform[:3])
        #     print(transform_batch[:3])
            # print(loss_func(pred_transform, transform_batch, points))

    return train_loss/n_train_steps

def test(dataset, batch_size):
    """
    Return loss and classification accuracy of the model on the test data
    """
    model.eval()
    test_loss, total = 0, 0

    # test_indices = dataset.split('train')[1][:1000]
    test_indices = dataset.split('train')[1]
    # test_indices = dataset.split('train2')[1][:64*100]
    n_test = len(test_indices)
    n_test_steps = n_test // batch_size

    ones = torch.Tensor(np.ones(batch_size)).to(device)

    with torch.no_grad():
        for step in tqdm(range(n_test_steps)):
            batch = dataset.get_item_list(test_indices[step*batch_size: (step+1)*batch_size])
            depth_image1 = batch["depth_image1"]
            depth_image2 = batch["depth_image2"]

            im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
            im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
            transform_batch = Variable(torch.from_numpy(batch["quaternion"])).to(device)
            pred_transform = model(im1_batch, im2_batch)
            total += transform_batch.size(0)

            obj_ids = batch["obj_id"]
            points_poses = batch["pose_matrix"][:,:3,:3]
            points = get_points(obj_ids, points_poses)
            # if config['loss'] == 'cosine':
            #     loss = loss_func(pred_transform, transform_batch, ones)

            sm_loss = loss_func2(pred_transform, transform_batch, points)
            test_loss += sm_loss.item()

    return test_loss/n_test_steps

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
    return args

if __name__ == '__main__':
    """Train on a dataset or generate a graph of the training and validation loss.
    Current Datasets: 
        72obj_occ: ~130,000 points, image 1 random rectangles, 57 train, 15 test
        uniform30, uniform45, uniform60: ~143,940 points, occlusions, angles sampled uniformly from 0-30,0-45,0-60
        564obj_250rot: 546(18 not available because of stable pose?) obj with more than 300 points
        best_scores: obj > 300 points, 257 obj, 500 rot. score >= 40. 128293
        546obj_dr: obj > 300 points. Occlusion is now w background pixels. ~130k. Initial pose from SO3
        best_scoresv2: occlusion is now w background pixels. 82 obj > 300 pts, 1800 rot, score >= 156.52. 163930. 16 obj in val
        546obj: obj > 300 points, 300 rot.  Initial pose has random translation. ~160k
        best_scoresv3: Initial pose translation. 82 obj > 300 pts, 2000 rot, score >= 156.52. 163930. 16 obj in val
        546objv2: No pose translation, no dr.
        546objv3: DR with pose sampling 0-45 degrees from stable pose
        best_scoresv4: No pose translation, no dr.
        best_scoresv5: DR with pose sampling 0-45 degrees from stable pose
        546objv4: DR with background, Translation(+-0.02,+-0.02,0-0.2), 45 degree from stable pose, 300 rot
        best_scoresv6: DR with background, Translation(+-0.02,+-0.02,0-0.2), 45 degree from stable pose, 300 rot
    """
    args = parse_args()
    config = YamlConfig(args.config)
    dataset_name = args.dataset + "/"
    args.dataset = os.path.join('/nfs/diskstation/projects/unsupervised_rbt', args.dataset)
    dataset = TensorDataset.open(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    prefix = "cos" if config['loss'] == "cosine" else "cos_sm"
    prefix += "_blk" + str(config['n_blocks']) + "_emb" + str(config["embed_dim"])
    prefix += "_reg" + str(config["reg"]) + "_drop" + str(config["dropout"])
    loss_history = "results/" + dataset_name + prefix + ".p"
    histdata = "results/" + dataset_name + prefix + "_histdata.txt"
    loss_plot_fname = "plots/" + dataset_name + prefix + "_loss.png"
    rot_plot_fname = "plots/" + dataset_name + prefix + "_rot.png"
    best_epoch_dir = "models/" + dataset_name + prefix + "_best.pt"
    print("fname prefix", prefix)

    model = ResNetSiameseNetwork(config['pred_dim'], config['n_blocks'], config['embed_dim'], config['dropout']).to(device)
    # model = InceptionSiameseNetwork(config['pred_dim']).to(device)

    # point_clouds = pickle.load(open("cfg/tools/data/point_clouds", "rb"))
    point_clouds = pickle.load(open("cfg/tools/data/point_clouds300", "rb"))
    scales = pickle.load(open("cfg/tools/data/scales", "rb"))

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=10**(-1 * config['reg']))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.9)
    loss_func = nn.CosineEmbeddingLoss()
    loss_func2 = ShapeMatchLoss()

    if not args.test:
        if not os.path.exists(args.dataset + "/splits/train"):
            obj_id_split = np.loadtxt("cfg/tools/data/train_split")
            val_indices = []
            for i in range(dataset.num_datapoints):
                if dataset.datapoint(i)["obj_id"] in obj_id_split:
                    val_indices.append(i)

            print("Created Train Split")
            dataset.make_split("train", train_pct=0.8, val_indices= val_indices)
        if not os.path.exists(args.dataset + "/splits/train2"):
            dataset.make_split("train2", train_pct=0.8)

        train_losses, test_losses = [], []
        min_loss = 100000
        # model.load_state_dict(torch.load("models/uniform30_1e7.pt"))

        for epoch in range(config['num_epochs']):

            train_loss = train(dataset, config['batch_size'], first = (epoch == 0))
            test_loss = test(dataset, config['batch_size'])
            scheduler.step()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print("Epoch %d, Train Loss = %f, Test Loss = %f" %
                  (epoch, train_loss, test_loss))
            pickle.dump({"train_loss": train_losses, "test_loss": test_losses,
                        }, open(loss_history, "wb"))
            # torch.save(model.state_dict(), final_epoch_dir)
            if test_loss < min_loss:
                torch.save(model.state_dict(), best_epoch_dir)
                min_loss = test_loss

    else:
        Plot_Loss(loss_history, loss_plot_fname)

        # model.load_state_dict(torch.load(final_epoch_dir))
        model.load_state_dict(torch.load(best_epoch_dir))
        # display_conv_layers(model)
        model.eval()
        test_loss, test_loss2, test_loss3, total = 0, 0, 0, 0

        # test_indices = dataset.split('train')[1][:1000]
        test_indices = dataset.split('train')[1]
        # test_indices = dataset.split('train2')[1]
        n_test = len(test_indices)
        batch_size = 1
        ones = torch.Tensor(np.ones(batch_size)).to(device)
        n_test_steps = n_test // batch_size

        true_quaternions = []
        pred_quaternions = []
        losses, angle_vs_losses = [], []
        with torch.no_grad():
            for step in tqdm(range(n_test_steps)):
                batch = dataset.get_item_list(test_indices[step*batch_size: (step+1)*batch_size])
                depth_image1 = batch["depth_image1"]
                depth_image2 = batch["depth_image2"]

                im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
                im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
                transform_batch = Variable(torch.from_numpy(batch["quaternion"])).to(device)
                pred_transform = model(im1_batch, im2_batch)
                # print("True Quaternions: {}, Predicted Quaternions: {}".format(transform_batch, pred_transform))
                total += transform_batch.size(0)

                loss = loss_func(pred_transform, transform_batch, ones).item()
                # angle_loss = np.arccos(1-loss) * 180 / np.pi * 2 # Don't use, always underestimates error.
                
                obj_ids = batch["obj_id"]
                points_poses = batch["pose_matrix"][:,:3,:3]
                points = get_points(obj_ids, points_poses)
                sm_loss = loss_func2(pred_transform, transform_batch, points).item()

                true_quaternions.extend(transform_batch.cpu().numpy())
                pred_quaternions.extend(pred_transform.cpu().numpy())

                true_quat = transform_batch.cpu().numpy()[0]
                angle = np.arccos(true_quat[3]) * 180 / np.pi * 2
                # print(true_quat[3], angle)
                losses.append(loss)
                angle_vs_losses.append([angle,loss,sm_loss])
                test_loss += loss
                test_loss2 += sm_loss
        np.savetxt(config['hist_data'], np.array(angle_vs_losses))
        mean_cosine_loss = test_loss/total
        mean_angle_loss = np.arccos(1-mean_cosine_loss)*180/np.pi*2
        Plot_Angle_vs_Loss(true_quaternions, losses, mean_angle_loss, rot_plot_fname)
        Plot_Small_Angle_Loss(true_quaternions, losses, mean_angle_loss)
        Plot_Axis_vs_Loss(true_quaternions, losses, mean_angle_loss)

        if args.worst_pred:
            biggest_losses = np.argsort(losses)[-5:-1]
            smallest_losses_idx = np.argsort(losses)
            smallest_losses = []
            for i in smallest_losses_idx:
                if true_quaternions[i][3] < 0.975:
                    smallest_losses.append(i)
                if len(smallest_losses) >= 5:
                    break
            Plot_Bad_Predictions(dataset, pred_quaternions, biggest_losses)
            Plot_Bad_Predictions(dataset, pred_quaternions, np.array(smallest_losses), "best")

        print("Mean Cosine loss is: ", test_loss/total)
        print("Mean Angle loss is: ", mean_angle_loss)
        print("Mean SM loss is: ", test_loss2/total)


