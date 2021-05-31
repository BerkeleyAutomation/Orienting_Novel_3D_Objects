from __future__ import division
from perception import DepthImage
from autolab_core import PointCloud, RigidTransform
import numpy as np
from tools.utils.stable_pose_utils import pointcloud
import trimesh
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins,1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins,1)
        return loss_1 + loss_2

def plot_PC(pcs):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    for pc in pcs:
        ax.scatter(pc[0], pc[1], pc[2])
    ax.view_init(elev=90, azim = 180)
    plt.show()

def demean(pc):
    #pc is 3xn
    return pc - np.mean(pc,1, keepdims=True)

def decenter(pc):
    #pc is 3xn
    max_bounds, min_bounds = np.max(pc,1, keepdims=True), np.min(pc,1, keepdims=True)
    bb_center = (max_bounds + min_bounds) / 2
    bb_center[2][0] = np.mean(pc,1)[2]
    return pc - bb_center

class PoseEstimator(object):
    def __init__(self):

        self.k, self.thresh, self.batch = 360, 0.00002, 360
        thetas = np.array([2 * np.pi * j / self.batch for j in range(self.batch)])
        offset = thetas[1] / (self.k // self.batch)

        R0 = np.array([[[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0          ,0              , 1]]  for theta in thetas])
        R0 = torch.Tensor(R0).to('cuda')

        R1 = np.array([[[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0          ,0              , 1]]  for theta in thetas + (offset * 1)])
        R1 = torch.Tensor(R1).to('cuda')

        R2 = np.array([[[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0          ,0              , 1]]  for theta in thetas + (offset * 2)])
        R2 = torch.Tensor(R2).to('cuda')

        R3 = np.array([[[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0          ,0              , 1]]  for theta in thetas + (offset * 3)])
        R3 = torch.Tensor(R3).to('cuda')

        R4 = np.array([[[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0          ,0              , 1]]  for theta in thetas + (offset * 4)])
        R4 = torch.Tensor(R4).to('cuda')

        self.R = [R0,R1,R2,R3,R4]

    def get_rotation(self, im1_batch, im2_batch): # input batch x 1 x height x width
        depth1, depth2 = im1_batch[0][0].cpu().numpy(), im2_batch[0][0].cpu().numpy()
        pc1, pc2 = pointcloud(depth1,300), pointcloud(depth2,300) #3xn1, 3xn2
        random_idx = np.random.randint(0,pc1.shape[1],size=500), np.random.randint(0,pc2.shape[1],size=500)
        pc1, pc2 = pc1[:,random_idx[0]], pc2[:,random_idx[1]]
        pc1 = torch.Tensor(demean(pc1)).float().to("cuda")
        pc2 = torch.Tensor(demean(pc2)).float().to("cuda")
        rot = self._is_match(pc1,pc2)
        pred_quat = RigidTransform.quaternion_from_axis_angle(np.array([0,0,1])*rot)
        quat = np.array([pred_quat[1],pred_quat[2],pred_quat[3],pred_quat[0]])
        tensor_quat = torch.from_numpy(quat).float().to('cuda').unsqueeze(0)
        return tensor_quat

    def _is_match(self, pc1, pc2):
        #pc1 and pc2 are 3xn_1, 3xn_2 point clouds
        # median_pc_distance(pc1)
        # print(pc1.mean(1), pc2.mean(1))
        min_dist = 999999
        pc1_batch = torch.unsqueeze(pc1, 0).repeat(self.batch, 1, 1)
        pc2_batch = torch.unsqueeze(pc2, 0).repeat(self.batch, 1, 1)

        for i in range(self.k//self.batch):
            pc1_R = torch.bmm(self.R[i],pc1_batch)
            chamfer_dist = ChamferLoss()#.to('cuda') # takes in two point clouds of nx3
            dist1 = chamfer_dist(pc1_R.transpose(2,1), pc2_batch.transpose(2,1)) #dist1 should be self.batch x n_1, dist2 batch x n_2
            cur_dist, best_dist = torch.min(dist1).item(), torch.argmin(dist1).item()
            if cur_dist < min_dist:
                min_dist = cur_dist
                rot = best_dist / self.batch * 2 * np.pi
        # print("Rotation around z is:", rot * 180 / np.pi)
        return rot + 1e-8

    def depth_to_world_seg(self, depth):
        point_cloud_cam = self._camera_intr.deproject(depth)
        point_cloud_cam.remove_zero_points()
        point_cloud_world = self._T_camera_world * point_cloud_cam
        seg_point_cloud_world, _ = point_cloud_world.box_mask(self._workspace)
        return seg_point_cloud_world

    def circle_segment(self, seg_pc, radius = 0.115):
        x_center, y_center = (np.max(seg_pc.x_coords) + np.min(seg_pc.x_coords)) / 2, (np.max(seg_pc.y_coords) + np.min(seg_pc.y_coords)) / 2
        distance = seg_pc.data[:2] - np.array([[x_center],[y_center]])
        radial_distance = np.linalg.norm(distance, axis = 0)
        mask = radial_distance < radius
        seg_pc._data = seg_pc._data[:,mask]
        return seg_pc

    def segment(self, depth, radial = False):
        seg_point_cloud_world = self.depth_to_world_seg(depth)
        if radial:
            seg_point_cloud_world = self.circle_segment(seg_point_cloud_world)
        
        seg_point_cloud_cam = self._T_camera_world.inverse() * seg_point_cloud_world
        depth_seg = self._camera_intr.project_to_image(seg_point_cloud_cam)
        return depth_seg

    def update_prev_pcs(self, prev_pcs):
        for cur_pc in prev_pcs:
            cur_pc = torch.Tensor(demean(cur_pc)).float().to("cuda")
            self.prev_pcs.append(cur_pc)
            self.prev_trans.append(np.zeros(3))
            self.next_stp += 1

    def get_latest_pc(self):
        latest_pc = self.prev_pcs[-1].cpu().detach().numpy()
        latest_trans = self.prev_trans[-1]
        n = latest_pc.shape[1]
        if n > 10000:
            points = np.random.choice(n, 10000, replace = False)
            latest_pc = latest_pc[:,points]
            num_points = 10000
        else:
            num_points = n
        return latest_pc, latest_trans, num_points
