import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation
from scipy.stats.morestats import anderson_ksamp
import torch
import torchvision
from autolab_core import YamlConfig, RigidTransform
from pyquaternion import Quaternion
import cv2
from mpl_toolkits.mplot3d import Axes3D
from perception import CameraIntrinsics, RgbdSensorFactory, Image, DepthImage
import os
import trimesh
from .plot_utils import *
from .rotation_utils import *
from pyrender import (Scene, IntrinsicsCamera, Mesh,
                      Viewer, OffscreenRenderer, RenderFlags, Node)
from sd_maskrcnn.envs import CameraStateSpace

def Aligned_Prism_Mesh_Base(mesh):
    prism_mesh = trimesh.creation.box(mesh.extents)
    obj_max, prism_max = np.max(mesh.vertices, axis = 0), np.max(prism_mesh.vertices, axis = 0)
    obj_min, prism_min = np.min(mesh.vertices, axis = 0), np.min(prism_mesh.vertices, axis = 0)
    obj_bb_center, prism_bb_center = (obj_max + obj_min) / 2, (prism_max + prism_min) / 2
    prism_center_alignment = obj_bb_center - prism_bb_center
    trans_matrix = np.eye(4)
    trans_matrix[:3,3] = prism_center_alignment
    prism_mesh.apply_transform(trans_matrix)
    return prism_mesh

def Percent_Fit_Surface(batch, predicted_quat):
    assert predicted_quat.shape[0] == 1
    predicted_quat = predicted_quat[0]
    object_id, pose_matrix = batch['obj_id'][0], batch['pose_matrix'][0]

    obj_id = 0
    for mesh_dir, mesh_filename in Load_Mesh_Path():
        obj_id += 1
        if obj_id != object_id:
            continue
        mesh = trimesh.load_mesh(os.path.join(mesh_dir, mesh_filename))
        pose_matrix[:3,3] = 0

        prism_mesh = Aligned_Prism_Mesh(mesh)
        
        mesh.apply_transform(pose_matrix)
        prism_mesh.apply_transform(pose_matrix)

        quat = predicted_quat
        quat[0], quat[1], quat[2], quat[3] = quat[3], quat[0], quat[1], quat[2]
        mesh.apply_transform(trimesh.transformations.quaternion_matrix(quat))

        quat = batch['quaternion'][0]
        quat[0], quat[1], quat[2], quat[3] = quat[3], quat[0], quat[1], quat[2]
        prism_mesh.apply_transform(trimesh.transformations.quaternion_matrix(quat))

        sampled_points = mesh.sample(500)
        contained_points = prism_mesh.contains(sampled_points)
        return np.sum(contained_points) / len(sampled_points)

def Percent_Fit_Intersection(batch, predicted_quat):
    assert predicted_quat.shape[0] == 1
    predicted_quat = predicted_quat[0]
    object_id, pose_matrix = batch['obj_id'][0], batch['pose_matrix'][0]

    obj_id = 0
    for mesh_dir, mesh_filename in Load_Mesh_Path():
        obj_id += 1
        if obj_id != object_id:
            continue
        
        mesh = Load_Scale_Mesh(mesh_dir, mesh_filename)
        pose_matrix[:3,3] = 0
        
        prism_mesh = Aligned_Prism_Mesh(mesh)
        
        mesh.apply_transform(pose_matrix)
        prism_mesh.apply_transform(pose_matrix)

        quat = predicted_quat
        quat[0], quat[1], quat[2], quat[3] = quat[3], quat[0], quat[1], quat[2]
        mesh.apply_transform(trimesh.transformations.quaternion_matrix(quat))

        quat = batch['quaternion'][0]
        quat[0], quat[1], quat[2], quat[3] = quat[3], quat[0], quat[1], quat[2]
        prism_mesh.apply_transform(trimesh.transformations.quaternion_matrix(quat))
        a,b,c = prism_mesh.extents

        intersection_volume = mesh.intersection(prism_mesh).volume
        # union_volume = mesh.union(prism_mesh).volume #crashes?
        mesh_volume = mesh.volume
        percent_fit = intersection_volume/mesh_volume
        # iou = intersection_volume/union_volume
        print("Percent Fit:", percent_fit)
        # print("IOU between mesh and goal bounding box:", iou)

        return percent_fit

def IOU_Points(batch,predicted_quat):
    predicted_quat = predicted_quat[0]
    object_id, pose_matrix = batch['obj_id'][0], batch['pose_matrix'][0]
    point_clouds = pickle.load(open("cfg/tools/data/point_clouds", "rb"))

    points = point_clouds[object_id]
    gt_rotation = Quaternion_to_Rotation(batch['quaternion'][0], np.array([0,0,0]))[:3,:3]
    pred_rotation = Quaternion_to_Rotation(predicted_quat, np.array([0,0,0]))[:3,:3]

    gt_points = gt_rotation.dot(points)
    pred_points = pred_rotation.dot(points)
    x,y,z = (np.max(gt_points, axis = 1) + np.min(gt_points, axis = 1)) / 2
    a,b,c = (np.max(pred_points, axis = 1) + np.min(pred_points, axis = 1)) / 2

    intersection = min((a,x)) * min((b,y)) * min((c,z))
    union = x*y*z + a*b*c - intersection
    iou = intersection/union
    if iou > 0 and iou <= 1:
        return iou
    else:
        return 1

def Intersection_Over_Union(batch, predicted_quat):
    assert predicted_quat.shape[0] == 1
    predicted_quat = predicted_quat[0]
    object_id, pose_matrix = batch['obj_id'][0], batch['pose_matrix'][0]

    obj_id = 0
    for mesh_dir, mesh_filename in Load_Mesh_Path():
        obj_id += 1
        if obj_id != object_id:
            continue
        
        mesh = Load_Scale_Mesh(mesh_dir, mesh_filename)
        pose_matrix[:3,3] = 0
        mesh.apply_transform(pose_matrix)
        mesh2 = mesh.copy()
        
        quat = predicted_quat
        quat[0], quat[1], quat[2], quat[3] = quat[3], quat[0], quat[1], quat[2]
        mesh.apply_transform(trimesh.transformations.quaternion_matrix(quat))
        x,y,z = mesh.extents

        quat = batch['quaternion'][0]
        quat[0], quat[1], quat[2], quat[3] = quat[3], quat[0], quat[1], quat[2]
        mesh2.apply_transform(trimesh.transformations.quaternion_matrix(quat))
        a,b,c = mesh2.extents
        intersection = min((a,x)) * min((b,y)) * min((c,z))
        union = x*y*z + a*b*c - intersection
        return intersection/union

def display_conv_layers(model):
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    with torch.no_grad():
        imshow(torchvision.utils.make_grid(model.resnet.conv1.weight))
