import numpy as np
import argparse
import os
import itertools
import open3d as o3d
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import pickle
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from autolab_core import YamlConfig, RigidTransform, BagOfPoints
from unsupervised_rbt import TensorDataset
from unsupervised_rbt.models import ResNetSiameseNetwork, InceptionSiameseNetwork
from unsupervised_rbt.losses.shapematch import ShapeMatchLoss
from perception import DepthImage, RgbdImage

from tools.data_gen_quat import create_scene
from tools.utils import *

import trimesh
from pyrender import (Scene, IntrinsicsCamera, Mesh,
                      Viewer, OffscreenRenderer, RenderFlags, Node)
import cv2
import visualization as vis
from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE;

def pointcloud(depth, fx, background=0.7999):
    """
    From: https://github.com/mmatl/pyrender/issues/14
    """
    fy = fx # assume aspectRatio is one.
    height = depth.shape[0]
    width = depth.shape[1]
    # mask = np.where(depth < background)
    mask = np.where(depth != 0)
    
    x = mask[1]
    y = mask[0]
    
    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = ((height - y).astype(np.float32) - height * 0.5) / height
    
    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = 0.8 - depth[y, x]
    # ones = np.ones(world_z.shape[0], dtype=np.float32)

    # points = np.vstack((world_x, world_y, world_z, ones)).T
    points = np.vstack((world_x, world_y, world_z)).T # Shape (n, 3)
    # print(points)

    return (points-np.mean(points,0)) * 1000

def predict(img1, img2):
    quats = []

    for i1, i2 in zip(img1,img2):
        # print(i1,i2)
        pc1 = pointcloud(i1, 300, background=0.1)
        pc2 = pointcloud(i2, 300, background=0.1)
        try:
            pred_transform_matrix = trimesh.registration.icp(pc1, pc2)[0]
        except:
            pred_transform_matrix = np.eye(4)
        # pred_transform_matrix = trimesh.registration.icp(pc1,pc2, initial = pred_transform_matrix)[0]
        pred_transform = Rotation_to_Quaternion(pred_transform_matrix)
        # pc1_trimesh = trimesh.points.PointCloud(pc1)
        # pc1_trimesh.apply_transform(pred_transform_matrix)
        # pc3 = pc1_trimesh.vertices

        # plt.imshow(i1)
        # Plot_ICP([pc1, pc2, pc3])
        # plt.imshow(i2)
        # Plot_ICP([pc2])

        quats.append(pred_transform)
    return np.array(quats)

def scale_pointcloud(pc):
    pc = pc - np.mean(pc, axis=0)
    pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc

def predict_goicp(img1, img2):
    def loadPointCloud(pc):
        pc = pc.astype(float)
        p3dlist = []
        for x,y,z in pc:
            pt = POINT3D(x,y,z)
            p3dlist.append(pt)
        return pc.shape[0], p3dlist

    quats = []
    for i1, i2 in zip(img1,img2):
        pc1 = scale_pointcloud(pointcloud(i1, 535))
        pc2 = scale_pointcloud(pointcloud(i2, 535))

        goicp = GoICP()
        goicp = GoICP()
        # rNode = ROTNODE()
        # tNode = TRANSNODE()
        # rNode.a = -3.1416
        # rNode.b = -3.1416
        # rNode.c = -3.1416
        # rNode.w = 6.2832
        # tNode.x = -0.5
        # tNode.y = -0.5
        # tNode.z = -0.5
        # tNode.w = 1.0

        goicp.MSEThresh = 0.001
        goicp.trimFraction = 0.4

        Nm, a_points = loadPointCloud(pc1)
        Nd, b_points = loadPointCloud(pc2)
        goicp.loadModelAndData(Nm, a_points, Nd, b_points)
        goicp.setDTSizeAndFactor(300, 2.0)
        # goicp.setInitNodeRot(rNode)
        # goicp.setInitNodeTrans(tNode)

        goicp.BuildDT()
        goicp.Register()
        # print(goicp.optimalRotation()) # A python list of 3x3 is returned with the optimal rotation
        # print(goicp.optimalTranslation())# A python list of 1x3 is returned with the optimal translation
        pred_transform_matrix = np.array(goicp.optimalRotation())
        pred_transform = Rotation_to_Quaternion(pred_transform_matrix)
        # pc1_trimesh = trimesh.points.PointCloud(pc1)
        # pc1_trimesh.apply_transform(pred_transform_matrix)
        # pc3 = pc1_trimesh.vertices

        # plt.imshow(i1)
        # Plot_ICP([pc1, pc2, pc3])
        # plt.imshow(i2)
        # Plot_ICP([pc2])

        quats.append(pred_transform)
    return np.array(quats)

def Plot_ICP(pointclouds):
    fig = plt.figure()
    ax = Axes3D(fig)
    for pc in pointclouds:
        ax.scatter(pc[:,0], pc[:,1], pc[:,2])
    ax.view_init(elev=90, azim=270)
    # plt.show()
    plt.savefig("plots/test.png")
    plt.close()

def orb_feature(img1, img2): #img1,2 depth images from pyrender
    img1, img2 = img1-img1.min(), img2-img2.min()
    img1, img2 = img1 / img1.max(), img2 / img2.max()
    img1, img2 = (img1 * 255).astype(np.uint8), (img2 * 255).astype(np.uint8)
    # print(img1.shape)
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    if len(kp1) == 0 or len(kp2) == 0:
        return np.array([]), np.array([])
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    print("No matches") and cv2.imwrite("plots/test.png", img1) if len(matches) == 0 else 0
    img1_coords, img2_coords = [], []
    def round_inwards(pt):
        pt = list(pt)
        pt[0] = np.floor(pt[0]) if pt[0] >= 64 else np.ceil(pt[0])
        pt[1] = np.floor(pt[1]) if pt[1] >= 64 else np.ceil(pt[1])
        return pt
    for match in matches:
        img1_coords.append(round_inwards(kp1[match.queryIdx].pt))
        img2_coords.append(round_inwards(kp2[match.trainIdx].pt))
    return np.array(img1_coords, np.int), np.array(img2_coords, np.int)

def pointcloud_old(depth, fx, keypoints):
    """
    From: https://github.com/mmatl/pyrender/issues/14
    """
    fy = fx # assume aspectRatio is one.
    height = depth.shape[0]
    width = depth.shape[1]

    x = keypoints[:,0]
    y = keypoints[:,1]
    
    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = (y.astype(np.float32) - height * 0.5) / height
    
    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = depth[y, x]
    # ones = np.ones(world_z.shape[0], dtype=np.float32)

    # points = np.vstack((world_x, world_y, world_z, ones)).T
    points = np.vstack((world_x, world_y, world_z)).T # Shape (n, 3)
    # print(points)

    return points

def predict_old(img1, img2):
    quats = []

    for i1, i2 in zip(img1,img2):
        # print(i1,i2)
        img1_kp, img2_kp = orb_feature(i1, i2)
        if len(img1_kp) >= 1:
            pc1 = pointcloud(i1, 535, img1_kp)
            pc2 = pointcloud(i2, 535, img2_kp)

            pred_transform_matrix = trimesh.registration.procrustes(pc1, pc2, reflection=False, 
                        translation=False, scale=False, return_cost=False)
            # pred_transform_matrix = trimesh.registration.icp(pc1,pc2, initial = pred_transform_matrix)[0]
            pred_transform = Rotation_to_Quaternion(pred_transform_matrix)
            quats.append(pred_transform)
        else:
            quats.append(np.array([0,0,0,1]))
    return np.array(quats)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = pcd

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def array_to_open3d(pc1, pc2):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pc1)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pc2)
    o3d.io.write_point_cloud("results/" + dataset_name + "pc1.ply", source)
    o3d.io.write_point_cloud("results/" + dataset_name + "pc2.ply", target)
    source = o3d.io.read_point_cloud("results/" + dataset_name + "pc1.ply")
    target = o3d.io.read_point_cloud("results/" + dataset_name + "pc2.ply")
    return source, target

def prepare_dataset(pc1, pc2, voxel_size):
    source, target = array_to_open3d(pc1, pc2)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    # draw_registration_result(source_down, target_down, np.identity(4))
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                        target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
    #         % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def predicta(img1, img2):
    quats = []

    for i1, i2 in zip(img1,img2):
        # print(i1,i2)
        pc1 = pointcloud(i1, 535)
        pc2 = pointcloud(i2, 535)

        voxel_size = 0.05  # means 5cm for the dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pc1, pc2, voxel_size)
        # source_down.scale(1 / np.max(source_down.get_max_bound() - source_down.get_min_bound()), center=source_down.get_center())
        # target_down.scale(1 / np.max(target_down.get_max_bound() - target_down.get_min_bound()), center=target_down.get_center())

        result_fgr = execute_fast_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)

        pred_transform_matrix = result_fgr.transformation
        # print(pred_transform_matrix)
        # draw_registration_result(source_down, target_down,result_fgr.transformation)
        pred_transform = Rotation_to_Quaternion(pred_transform_matrix)
        # print(pred_transform)
        pc1_trimesh = trimesh.points.PointCloud(pc1)
        pc1_trimesh.apply_transform(pred_transform_matrix)
        pc3 = pc1_trimesh.vertices

        # plt.imshow(i1)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # # ax.scatter(pc1[:,0], pc1[:,1], pc1[:,2])
        # ax.scatter(pc2[:,0], pc2[:,1], pc2[:,2])
        # ax.scatter(pc3[:,0], pc3[:,1], pc3[:,2])
        # ax.view_init(elev=90, azim=270)
        # plt.show()

        # plt.imshow(i2)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(pc2[:,0], pc2[:,1], pc2[:,2])
        # ax.view_init(elev=90, azim=270)
        # plt.show()

        quats.append(pred_transform)
    return np.array(quats)

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
        546objv5: DR with background, Translation(+-0.01,+-0.01,0-0.05), 45 degree from stable pose, 300 rot, z buffer (0.4,2)
    """
    args = parse_args()
    config = YamlConfig(args.config)
    dataset_name = args.dataset + "/"
    args.dataset = os.path.join('/nfs/diskstation/projects/unsupervised_rbt', args.dataset)
    dataset = TensorDataset.open(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    prefix = "icp"
    loss_history = "results/" + dataset_name + prefix + ".p"
    histdata = "results/" + dataset_name + prefix + "_histdata.txt"
    loss_plot_fname = "plots/" + dataset_name + prefix + "_loss.png"
    rot_plot_fname = "plots/" + dataset_name + prefix + "_rot.png"
    best_epoch_dir = "models/" + dataset_name + prefix + ".pt"
    print("fname prefix", prefix)

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

    point_clouds = pickle.load(open("cfg/tools/data/point_clouds", "rb"))
    # point_clouds = pickle.load(open("cfg/tools/data/point_clouds300", "rb"))
    scales = pickle.load(open("cfg/tools/data/scales", "rb"))

    loss_func = nn.CosineEmbeddingLoss()
    loss_func2 = ShapeMatchLoss()

    test_loss, test_loss2, test_loss3, total = 0, 0, 0, 0

    # test_indices = dataset.split('train')[1][:100]
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
            im1_batch = batch["depth_image1"]
            im2_batch = batch["depth_image2"]
            transform_batch = Variable(torch.from_numpy(batch["quaternion"])).to(device)

            pred_transform = predict(im1_batch[:,0], im2_batch[:,0]) # shape is (batch_size, 1, 128,128)
            # pred_transform = predict_goicp(im1_batch[:,0], im2_batch[:,0]) # shape is (batch_size, 1, 128,128)
            pred_transform = Variable(torch.from_numpy(pred_transform)).float().to(device)
            # print("True Quaternions: {}, Predicted Quaternions: {}".format(transform_batch, pred_transform))
            total += transform_batch.size(0)
            loss = loss_func(pred_transform, transform_batch, ones).item()

            obj_ids = batch["obj_id"]
            points_poses = batch["pose_matrix"][:,:3,:3]
            points = get_points(obj_ids, points_poses, point_clouds, scales, device)
            sm_loss = loss_func2(pred_transform, transform_batch, points).item()
            # print(sm_loss)

            true_quaternions.extend(transform_batch.cpu().numpy())
            pred_quaternions.extend(pred_transform.cpu().numpy())

            true_quat = transform_batch.cpu().numpy()[0]
            angle = np.arccos(true_quat[3]) * 180 / np.pi * 2
            # print(true_quat[3], angle)
            # losses.append(loss)

            losses.append(sm_loss)
            angle_vs_losses.append([angle,loss,sm_loss])
            test_loss += loss
            test_loss2 += sm_loss
    np.savetxt(histdata, np.array(angle_vs_losses))
    mean_cosine_loss = test_loss/total
    mean_angle_loss = np.arccos(1-mean_cosine_loss)*180/np.pi*2
    Plot_Angle_vs_Loss(angle_vs_losses, rot_plot_fname, "shapematch")
    # Plot_Angle_vs_Loss(angle_vs_losses, rot_plot_fname, "cosine")
    # Plot_Small_Angle_Loss(true_quaternions, losses, mean_angle_loss)
    # Plot_Axis_vs_Loss(true_quaternions, losses, mean_angle_loss)

    # if args.worst_pred:
    #     biggest_losses = np.argsort(losses)[-5:-1]
    #     smallest_losses_idx = np.argsort(losses)
    #     smallest_losses = []
    #     for i in smallest_losses_idx:
    #         if true_quaternions[i][3] < 0.975:
    #             smallest_losses.append(i)
    #         if len(smallest_losses) >= 5:
    #             break
    #     Plot_Bad_Predictions(dataset, pred_quaternions, biggest_losses)
    #     Plot_Bad_Predictions(dataset, pred_quaternions, np.array(smallest_losses), "best")

    print("Mean Cosine loss is: ", test_loss/total)
    print("Mean Angle loss is: ", mean_angle_loss)
    print("Mean SM loss is: ", test_loss2/total)


