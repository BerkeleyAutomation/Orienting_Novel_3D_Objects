from autolab_core import YamlConfig, RigidTransform, TensorDataset
from scipy.spatial.transform import Rotation
import os
import time
import torch
import numpy as np
import trimesh
import itertools
import sys
import argparse
import pyrender
import matplotlib.pyplot as plt
import random
from termcolor import colored
import pickle
from .utils import *
from tools.chamfer_distance import ChamferDistance
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, ConfusionMatrixDisplay

k, thresh, trans, batch = 7200*3, 0.00001, 0.01, 7200*3
thetas = [2 * np.pi * j / batch for j in range(batch)]
R = np.array([[[np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0          ,0              , 1]]  for theta in thetas])
R = torch.Tensor(R).to('cuda')

def is_match(pc1, pc2):
    #pc1 and pc2 are 3xn point clouds
    # median_pc_distance(pc1)
    # print(pc1.mean(1), pc2.mean(1))
    d = (pc1.shape[1] * 4.75) // 5
    highest = 0
    pc1_batch = torch.unsqueeze(pc1, 0).repeat(batch, 1, 1)
    pc2_batch = torch.unsqueeze(pc2, 0).repeat(batch, 1, 1)

    for i in range(k//batch):
        # T = np.array([[(np.random.random()*trans*2) - trans, (np.random.random()*trans*2) - trans,0] for theta in thetas])
        # T = torch.Tensor(T[:,:,None]).to('cuda')
        pc1_R = torch.bmm(R,pc1_batch) #+ T
        chamfer_dist = ChamferDistance().to('cuda') # takes in two point clouds of nx3
        dist1, dist2 = chamfer_dist(pc1_R.transpose(2,1), pc2_batch.transpose(2,1)) #dist1 should be batch x n
        batch_inliers = torch.sum(dist1 < thresh, 1)
        num_inliers, best_inliers = torch.max(batch_inliers).item(), torch.argmax(batch_inliers).item()
        # Plot_PC([pc1_R.cpu().numpy().T,pc2.cpu().numpy().T], "plots/RANSAC{}.png".format(i)) if i < 10 else 0
        if num_inliers / pc1.shape[1] > highest:
            highest = num_inliers / pc1.shape[1]
        #     # Plot_PC([pc1_R.cpu().numpy()[best_inliers].T,pc2.cpu().numpy().T], "plots/best_inliers.png")

        # for n, pc in enumerate(pc1_R.cpu().numpy()[::72*10]):
        #     Plot_PC([pc.T,pc2.cpu().numpy().T], "plots/pc{}.png".format(n))

        if num_inliers > d:
            # print("Inliers on iteration", i, "with num inliers:", num_inliers, "and d is", d)
            # print("Breaking on translation",T[best_inliers].cpu().numpy()[:2,0])
            # Plot_PC([pc1_R.cpu().numpy()[best_inliers].T,pc2.cpu().numpy().T], "plots/RANSAC.png")
            break
    # print("Highest percent of inliers:", round(highest * 100,2))
    return (True, round(highest * 100,2)) if num_inliers > d else (False, round(highest * 100,2))

def plot_precision_recall_roc():
    results = np.load("results/pose_est/results.npy")
    analyze_results(results, save=False)

    precision, recall, thresholds = precision_recall_curve(results[:,2], results[:,4]/100)
    plt.plot(recall,precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.savefig("plots/precision_recall.png")
    plt.close()

    fpr, tpr, thresholds = roc_curve(results[:,2], results[:,4]/100)
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("plots/roc.png")
    plt.close()

    cm = confusion_matrix(results[:,2], results[:,3], normalize='true')
    ax = plt.figure(figsize=(9,6)).gca()
    disp = ConfusionMatrixDisplay(cm,["Different Stable Pose", "Same Stable Pose"])
    disp.plot(cmap='GnBu', ax = ax)
    plt.savefig("plots/confusion.png")
    plt.close()

    cm = stp_confusion(results)
    ax = plt.figure(figsize=(9,6)).gca()
    # print(cm)
    disp = ConfusionMatrixDisplay(cm, display_labels = range(cm.shape[0]))
    disp.plot(cmap='GnBu', ax = ax)
    ax.set(ylabel="Stable Pose",
               xlabel="Stable Pose")
    plt.savefig("plots/stp_confusion.png")
    plt.close()

def stp_confusion(results):
    for r in results:
        if r[0] < r[1]:
            r[0], r[1] = r[1], r[0]
    num_stp = int(results[:,0].max()) + 1
    cm = np.zeros((num_stp,num_stp))
    for i in range(num_stp):
        for j in range(i+1):
            relevant_results = results[np.logical_and(results[:,0]==i, results[:,1] == j)]
            cm[i,j] = np.mean(relevant_results[:,2]==relevant_results[:,3])
            cm[j,i] = cm[i,j]
            # if i == 12 and j == 6:
            #     print(relevant_results[:,4])
    return cm

def analyze_results(results, save = True):
    results = np.array(results)
    true_positives = results[results[:,2]==True]
    true_negatives = results[results[:,2]==False]
    for i in [80,85,90,95,98,99]:
        print("For Threshold", i)
        pred_positives = results[results[:,4]>=i]
        print("Accuracy is", np.sum(results[:,2]==(results[:,4]>=i))/results.shape[0])
        print("Precision is", np.sum(pred_positives[:,2]==(pred_positives[:,4]>=i))/pred_positives.shape[0])
        print("Recall is", np.sum(true_positives[:,2]==(true_positives[:,4]>=i))/true_positives.shape[0])
    # print(true_positives[true_positives[:,3]==False])
    np.save("results/pose_est/results.npy", results) if save else 0
    i,j = np.argmax(true_negatives[:,4]), np.argmin(true_positives[:,4])
    print("Highest score for Different Stable Pose is", true_negatives[i,4], "on stable poses", true_negatives[i,0], true_negatives[i,1])
    print("Lowest score for Same Stable Pose is", true_positives[j,4], "on stable poses", true_positives[j,0], true_positives[j,1])

def pointcloud(depth, fx=600):
    """
    From: https://github.com/mmatl/pyrender/issues/14
    """
    fy = fx # assume aspectRatio is one.
    height = depth.shape[0]
    width = depth.shape[1]
    mask = np.where(depth != 0)
    
    x = mask[1]
    y = mask[0]
    
    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = ((height - y).astype(np.float32) - height * 0.5) / height
    
    world_x = normalized_x * depth[y, x] / fx * 100
    world_y = normalized_y * depth[y, x] / fy * 100
    world_z = 0.8 - depth[y, x]
    # ones = np.ones(world_z.shape[0], dtype=np.float32)

    # points = np.vstack((world_x, world_y, world_z, ones)).T
    points = np.vstack((world_x, world_y, world_z)) # Shape (3, n)
    # print(points)

    return points

def demean(pc):
    #pc is 3xn
    return pc - pc.mean(1)[:,None]

def median_pc_distance(pc):
    #pc is 3xn
    pc = torch.transpose(pc, 0, 1) #nx3
    pc1 = pc.unsqueeze(0) # 1xnx3
    pc2 = pc.unsqueeze(1) # nx1x3
    pc1 = pc1.repeat(pc.shape[0],1,1) # nxnx3
    pc2 = pc2.repeat(1,pc.shape[0],1) # nxnx3

    squared_diffs = torch.abs(pc1-pc2) # nxnx3
    squared_diffs = torch.sum(squared_diffs, 2, keepdim=True).squeeze(2) # nxn
    min_diffs, _ = torch.min(squared_diffs, 1) # n
    print("mean distance", torch.mean(min_diffs))
    print("median distance", torch.median(min_diffs))
    return torch.mean(min_diffs)

