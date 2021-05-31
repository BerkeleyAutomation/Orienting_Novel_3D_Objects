'''
Train self-supervised task (rotation prediction) task; current good dataset to use is 872objv2; 
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
from unsupervised_rbt.models import ResNetSiameseNetwork, Se3TrackNet
from unsupervised_rbt.losses.shapematch import ShapeMatchLoss
# from perception import DepthImage, RgbdImage
from tools.utils import *

import time 

def Make_Train_Splits(dataset_path, dataset):
    """Split the data into training and validation..
    Split 'train' adds all objects from a file to the validation split and leaves the rest in training
    Split 'train2' randomly samples 80% of the data for training and 20% for validation."""
    if not os.path.exists(dataset_path + "/splits/train"):
        print("Creating Train Split")
        obj_id_split = np.loadtxt("cfg/tools/data/train_split_872")
        # obj_id_split = np.loadtxt("cfg/tools/data/train_split_100")
        val_indices = []
        for i in tqdm(range(dataset.num_datapoints)):
            if dataset.datapoint(i)["obj_id"] in obj_id_split:
                val_indices.append(i)
        dataset.make_split("train", train_pct=0.8, val_indices=val_indices)
    if not os.path.exists(dataset_path + "/splits/train2"):
        dataset.make_split("train2", train_pct=0.8)

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
        depth_image1 = Quantize(batch["depth_image1"],demean=config['demean'])
        depth_image2 = Quantize(batch["depth_image2"],demean=config['demean'])
        # depth_image1 = batch["depth_image1"]
        # depth_image2 = batch["depth_image2"]

        im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
        im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
        true_quaternions = Variable(torch.from_numpy(batch["quaternion"])).to(device)
        obj_ids = batch["obj_id"]
        points_poses = batch["pose_matrix"][:,:3,:3]
        points = point_cloud_fn(obj_ids, points_poses, point_clouds, scales, device)

        pred_quaternions = model(im1_batch, im2_batch)
        if config['loss'] == 'cosine' or first:
            loss = loss_func(pred_quaternions, true_quaternions, ones)
            sm_loss = loss_func2(pred_quaternions, true_quaternions, points).item()
        else:
            loss = loss_func2(pred_quaternions, true_quaternions, points)
            sm_loss = loss.item()

        # loss = loss / 3
        # loss.backward()
        # if step % 3 == 2 or step == n_train_steps - 1:
        #     optimizer.step()
        #     optimizer.zero_grad()

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

    # test_indices = dataset.split('train')[1][:10000]
    test_indices = dataset.split('train')[1][::2]
    # test_indices = dataset.split('train2')[1][:64*100]
    n_test = len(test_indices)
    n_test_steps = n_test // batch_size

    ones = torch.Tensor(np.ones(batch_size)).to(device)

    with torch.no_grad():
        for step in tqdm(range(n_test_steps)):
            batch = dataset.get_item_list(test_indices[step*batch_size: (step+1)*batch_size])
            depth_image1 = Quantize(batch["depth_image1"],demean=config['demean'])
            depth_image2 = Quantize(batch["depth_image2"],demean=config['demean'])
            # depth_image1 = batch["depth_image1"]
            # depth_image2 = batch["depth_image2"]

            im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
            im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
            true_quaternions = Variable(torch.from_numpy(batch["quaternion"])).to(device)
            pred_quaternions = model(im1_batch, im2_batch)
            total += true_quaternions.size(0)

            obj_ids = batch["obj_id"]
            points_poses = batch["pose_matrix"][:,:3,:3]
            points = point_cloud_fn(obj_ids, points_poses, point_clouds, scales, device)
            # if config['loss'] == 'cosine':
            #     loss = loss_func(pred_quaternions, true_quaternions, ones)

            sm_loss = loss_func2(pred_quaternions, true_quaternions, points)
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
        872obj: Translation +-(0.01,0.01,0.18-0.23), 128 rot, Zeroed DR, SO3, z buffer (0.4,2), Scaled mesh sizes 0.2-0.25
        best_scoresv6: 100 objects, 1500 rot, Translation as above, SO3, Zeroed DR, score >= 156.52
        872objv2: Above, but with 512 rot 
        872objv3: v2 but new scene, z (0.2,1.5), cropping 10% slack on center, mesh size 0.07-0.1, tr (0.05,0.07,0.25,0.4)
        872objv3: v2 but new scene, z (0.2,1.5), cropping 5-25% slack on center +-5, mesh size 0.07-0.2, tr (0.05,0.07,0.2,0.4)
    """
    args = parse_args()
    config = YamlConfig(args.config)
    dataset_name = args.dataset + "/"
    args.dataset = os.path.join('/nfs/diskstation/projects/unsupervised_rbt', args.dataset)
    dataset = TensorDataset.open(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    prefix = "cos" if config['loss'] == "cosine" else "cos_sm"
    prefix += "_blk" + str(config['n_blocks'])# + "_emb" + str(config["embed_dim"])
    prefix += "_reg" + str(config["reg"])# + "_drop" + str(config["dropout"])
    loss_history = "results/" + dataset_name + prefix + ".p"
    histdata = "results/" + dataset_name + prefix + "_histdata.txt"
    loss_plot_fname = "plots/" + dataset_name + prefix + "_loss.png"
    rot_plot_fname = "plots/" + dataset_name + prefix + "_rot"
    fit_plot_fname = "plots/" + dataset_name + prefix + "_fit"
    best_epoch_dir = "models/" + dataset_name + prefix + ".pt"
    print("fname prefix", prefix)
    make_dirs(dataset_name)

    model = ResNetSiameseNetwork(split_resnet=config['split_resnet']).to(device)
    # model = Se3TrackNet().to(device)

    point_clouds = pickle.load(open("cfg/tools/data/surface_pc_1000", "rb"))
    # point_clouds = pickle.load(open("cfg/tools/data/point_clouds", "rb"))
    point_cloud_fn = get_points_random_obj if config['shuffled'] else get_points_single_obj
    scales = pickle.load(open("cfg/tools/data/scales", "rb"))
    
    if config['shuffled']:
        point_clouds_arr = np.zeros((len(point_clouds),3,1000))
        for obj_id in range(1,len(point_clouds_arr)+1):
            point_clouds_arr[obj_id-1] = point_clouds[obj_id] / scales[obj_id] * 10
        point_cloud_fn = get_points_numpy
        point_clouds = point_clouds_arr
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=10**(-1 * config['reg']))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.95)
    loss_func = nn.CosineEmbeddingLoss()
    loss_func2 = ShapeMatchLoss()

    if not args.test:
        Make_Train_Splits(args.dataset, dataset)
        train_losses, test_losses = [], []
        min_loss = 100000

        for epoch in range(config['num_epochs']):
            train_loss = train(dataset, config['batch_size'], first = (epoch == 0))
            test_loss = test(dataset, config['batch_size'])
            scheduler.step()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(("Epoch %d, Train Loss = %f, Test Loss = %f" %
                  (epoch, train_loss, test_loss)) + " for " + prefix)
            pickle.dump({"train_loss": train_losses, "test_loss": test_losses,
                        }, open(loss_history, "wb"))
            # torch.save(model.state_dict(), final_epoch_dir)
            if test_loss < min_loss:
                torch.save(model.state_dict(), best_epoch_dir)
                min_loss = test_loss
            Plot_Loss(loss_history, loss_plot_fname)
    else:
        Make_Train_Splits(args.dataset, dataset)
        # Plot_Loss(loss_history, loss_plot_fname)

        if config['load_orienting_model']:
            print("Loading model that is trained from orienting task")
            model.load_state_dict(torch.load("models/872objv2/cos_sm_blk1_emb1024_reg9_drop4.pt"))
        else:
            # model.load_state_dict(torch.load(final_epoch_dir))
           model.load_state_dict(torch.load(best_epoch_dir))

        # display_conv_layers(model)
        model.eval()
        test_loss, test_loss2, total_fit, total = 0, 0, 0, 0

        pose_estim = PoseEstimator()

        test_indices = dataset.split('train')[1][::4]
        # test_indices = dataset.split('train2')[1]
        n_test = len(test_indices)
        batch_size = 1
        ones = torch.Tensor(np.ones(batch_size)).to(device)
        n_test_steps = n_test // batch_size

        true_quaternions_list, pred_quaternions_list = [], []
        losses, fit_losses, angle_vs_losses = [], [], []
        with torch.no_grad():
            for step in tqdm(range(n_test_steps)):
                batch = dataset.get_item_list(test_indices[step*batch_size: (step+1)*batch_size])
                # depth_image1 = batch["depth_image1"]
                # depth_image2 = batch["depth_image2"]
                depth_image1 = Quantize(batch["depth_image1"],demean=config['demean'])
                depth_image2 = Quantize(batch["depth_image2"],demean=config['demean'])

                im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
                im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
                transform_batch = Variable(torch.from_numpy(batch["quaternion"])).to(device)

                pred_transform = pose_estim.get_rotation(im1_batch, im2_batch)
                # pred_transform = model(im1_batch, im2_batch)
                # print("True Quaternions: {}, Predicted Quaternions: {}".format(transform_batch, pred_transform))
                total += transform_batch.size(0)

                cosine_loss = loss_func(pred_transform, transform_batch, ones).item()
                # angle_loss = error2angle(cosine_loss) # Don't use, always underestimates error.

                obj_ids = batch["obj_id"]
                points_poses = batch["pose_matrix"][:,:3,:3]
                points = point_cloud_fn(obj_ids, points_poses, point_clouds, scales, device)
                sm_loss = loss_func2(pred_transform, transform_batch, points).item()

                true_quaternions_list.extend(transform_batch.cpu().numpy())
                pred_quaternions_list.extend(pred_transform.cpu().numpy())

                # random_quat = Generate_Quaternion_SO3()[None,:]

                # start_time = time.time()
                fit_loss = 1 - Percent_Fit(batch, pred_transform.cpu().numpy())                
                # fit_loss = 1 - Percent_Fit(batch, random_quat)
                # fit_loss = 0.99
                
                # print(time.time() - start_time)

                true_quat = transform_batch.cpu().numpy()[0]
                angle = np.arccos(true_quat[3]) * 180 / np.pi * 2
                # print(true_quat[3], angle)

                angle_vs_losses.append([obj_ids[0],angle,cosine_loss,sm_loss,fit_loss])
                test_loss += cosine_loss
                test_loss2 += sm_loss
                total_fit += 1 - fit_loss
        np.savetxt(histdata, np.array(angle_vs_losses))
        mean_cosine_loss = test_loss/total
        mean_angle_loss = np.arccos(1-mean_cosine_loss)*180/np.pi*2
        Plot_Angle_vs_Loss(angle_vs_losses, rot_plot_fname, 'shapematch', max_angle=config['max_angle'])
        Plot_Angle_vs_Loss(angle_vs_losses, fit_plot_fname, 'fit', max_angle=config['max_angle'])
        # Plot_Eccentricity_vs_Fit(angle_vs_losses, rot_plot_fname, 0)
        # Plot_Eccentricity_vs_Fit(angle_vs_losses, rot_plot_fname, 10)
        Plot_Eccentricity_vs_Fit(angle_vs_losses, rot_plot_fname, 20)

        # Plot_Angle_vs_Loss(angle_vs_losses, rot_plot_fname, 'cosine')
        # Plot_Small_Angle_Loss(angle_vs_losses)
        # Plot_Axis_vs_Loss(true_quaternions_list, losses, mean_angle_loss)

        if args.worst_pred:
            Plot_Bad_Predictions(dataset, pred_quaternions_list, true_quaternions_list, angle_vs_losses, test_indices)
            Plot_Bad_Predictions(dataset, pred_quaternions_list, true_quaternions_list, angle_vs_losses, test_indices, "best")

        print("Mean Cosine loss is: ", test_loss/total)
        print("Mean Angle loss is: ", mean_angle_loss)
        print("Mean SM loss is: ", test_loss2/total)
        print("Mean Percent Fit is: ", total_fit/total)


