from __future__ import division
from perception import DepthImage
from autolab_core import PointCloud, RigidTransform
import numpy as np
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
    def __init__(self, camera_intr, T_camera_world, workspace, prev_pcs=None):
        self.prev_pcs, self.prev_trans, self.next_stp = [], [], 0
        self._camera_intr = camera_intr
        self._T_camera_world = T_camera_world
        self._workspace = workspace

        self.sim_to_real = False
        self.update_prev_pcs(prev_pcs) if prev_pcs else 0

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

    def get_pose(self, img, radial = False):
        depth1 = img.depth

        pc1 = self.depth_to_world_seg(depth1)
        if radial:
            pc1 = self.circle_segment(pc1)
        pc1 = pc1.data

        if np.prod(pc1.shape) < 600:
            print pc1.shape
            print "No object detected or less than 200 points in point cloud"
            return None, None, None

        max_bounds, min_bounds = np.max(pc1,1), np.min(pc1,1)
        # print "max bounds:", max_bounds
        # print "min bounds:", min_bounds
        # sys.exit()
        if max_bounds[0] > 0.480 or max_bounds[1] > 0.162 or min_bounds[0] < 0.272 or min_bounds[1] < -0.060:
            print "max bounds:", max_bounds
            print "min bounds:", min_bounds
            return None, None, None

        # trans1 = (max_bounds + min_bounds) / 2
        trans1 = np.mean(pc1,1)

        trans1[2] = 0

        best_estimate = [0,0,0]
        # plot_PC([demean(pc1), self.prev_pcs[1].cpu().detach().numpy()])
        pc1 = torch.Tensor(demean(pc1)).float().to("cuda")
        matched = False
        for i, prev_pc in enumerate(self.prev_pcs):
            result, rot, highest_inliers, min_dist = self._is_match(pc1, prev_pc)
            print('Inliers are', highest_inliers, 'Min dist is', min_dist)
            return_trans = RigidTransform.rotation_from_axis_and_origin(np.array([0,0,-1]), self.prev_trans[i], rot, to_frame='world')
            return_trans.translation += trans1 - self.prev_trans[i]
            if highest_inliers > best_estimate[1]:
                best_estimate = [i,highest_inliers, return_trans]
            if result:
                est_pose = i
                matched = True
                print('We think rotation is:', round(rot * 180 / np.pi,2), 'Degrees and Translation is:', trans1-self.prev_trans[i])
                break
        if not matched:
            if self.sim_to_real:
                est_pose, _, return_trans = best_estimate
            else:
                est_pose = self.next_stp
                self.prev_pcs.append(pc1)
                self.prev_trans.append(trans1)
                self.next_stp += 1
                return_trans = RigidTransform(from_frame = 'world', to_frame = 'world')

        return est_pose, self.segment(depth1), return_trans
        
    def _is_match(self, pc1, pc2):
        #pc1 and pc2 are 3xn_1, 3xn_2 point clouds
        # median_pc_distance(pc1)
        # print(pc1.mean(1), pc2.mean(1))
        d = (pc1.shape[1] * 4.1) // 5
        highest, min_dist = 0, 999999
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
        # print("Highest percent of inliers:", round(highest * 100,2))
        return rot

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
