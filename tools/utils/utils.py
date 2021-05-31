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
from perception import CameraIntrinsics

def Crop_Image(img):
    if img.shape[1] == 128:
        return img
    assert img.shape[0] == 772 and img.shape[1] == 1032
    default_i, default_j = 386, 516

    ind = np.where(img < 0.7999)
    mean_i, mean_j = int(np.mean(ind[0])), int(np.mean(ind[1]))

    height, width = np.max(ind[0])- np.min(ind[0]), np.max(ind[1])- np.min(ind[1])
    crop_factor = ((np.random.rand() * 2) + 10.5)
    height, width = (height*crop_factor) // 10, (width*crop_factor) // 10
    #TODO might be performing badly because image 1 will have different crop from image 2
    height, width = int(max((height,128,width))), int(max((width,128,height)))
    # print("Crop size is ", height, width) 

    max_i, max_j, min_i, min_j = np.max(ind[0]), np.max(ind[1]), np.min(ind[0]), np.min(ind[1])
    #TODO might be performing badly because image 1 will have different center from image 2
    center_i, center_j = (max_i+min_i)//2, (max_j+min_j)//2
    center_i, center_j = center_i + int(np.random.randint(-5,6)), center_j + int(np.random.randint(-5,6))
    assert center_i >= height // 2 and center_j >= width // 2, "Center is : {},{}, Height and Width are: {} {}".format(center_i, center_j, height, width)
    img_crop = img[center_i-height//2:center_i+height//2,center_j-width//2:center_j+width//2]
    img_crop = cv2.resize(img_crop,(128,128), interpolation = cv2.INTER_NEAREST)
    return img_crop

def create_scene_real(config=None):
    """Create scene for taking depth images.
    """
    if config is None:
        config = YamlConfig(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..', '..',
                                           'cfg/tools/data_gen_quat.yaml'))

    scene = Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
    renderer = OffscreenRenderer(viewport_width=1, viewport_height=1)
    # initialize camera and renderer
    phoxi_intr = CameraIntrinsics.load("/nfs/diskstation/calib/phoxi/phoxi.intr")
    T_camera_world = RigidTransform.load(os.path.join("/nfs/diskstation/calib/phoxi/phoxi_to_world.tf"))
    camera = IntrinsicsCamera(phoxi_intr.fx, phoxi_intr.fy,
                              phoxi_intr.cx, phoxi_intr.cy,
                              znear = 0.2, zfar = 1.5)
    renderer.viewport_width = phoxi_intr.width
    renderer.viewport_height = phoxi_intr.height

    pose_m = np.array([
        [0.0, -1.0,  0.0, 0.0],
        [-1.0, 0.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.8],
        [0.0,  0.0,  0.0, 1.0]
      ])
    pose_m[:, 1:3] *= -1.0

    scene.add(camera, pose=pose_m, name=phoxi_intr.frame)
    scene.main_camera_node = next(iter(scene.get_nodes(name=phoxi_intr.frame)))

    # Add Table
    table_mesh = trimesh.load_mesh(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '../'
            '../data/objects/plane/plane.obj',
        )
    )
    table_tf = RigidTransform.load(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '../',
            '../data/objects/plane/pose.tf',
        )
    )
    table_mesh.visual = table_mesh.visual.to_color()
    table_mesh.visual.vertex_colors = [[0 for c in r] for r in table_mesh.visual.vertex_colors]
    table_mesh = Mesh.from_trimesh(table_mesh)
    table_node = Node(mesh=table_mesh, matrix=table_tf.matrix)
    scene.add_node(table_node)

    # scene.add(Mesh.from_trimesh(table_mesh), pose=table_tf.matrix, name='table')
    return scene, renderer

def make_dirs(dataset_name):
    if not os.path.exists("results/" + dataset_name):
        os.makedirs("results/" + dataset_name)
    if not os.path.exists("plots/" + dataset_name):
        os.makedirs("plots/" + dataset_name)
    if not os.path.exists("models/" + dataset_name):
        os.makedirs("models/" + dataset_name)

def create_scene(config=None):
    """Create scene for taking depth images.
    """
    if config is None:
        config = YamlConfig(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..', '..',
                                           'cfg/tools/data_gen_quat.yaml'))

    scene = Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
    renderer = OffscreenRenderer(viewport_width=1, viewport_height=1)
    # initialize camera and renderer
    cam = CameraStateSpace(config['state_space']['camera']).sample()

    # If using older version of sd-maskrcnn
    # camera = PerspectiveCamera(cam.yfov, znear=0.05, zfar=3.0,
    #                                aspectRatio=cam.aspect_ratio)
    camera = IntrinsicsCamera(cam.intrinsics.fx, cam.intrinsics.fy,
                              cam.intrinsics.cx, cam.intrinsics.cy,
                              znear = 0.4, zfar = 2)
    renderer.viewport_width = cam.width
    renderer.viewport_height = cam.height

    pose_m = cam.pose.matrix.copy()
    pose_m[:, 1:3] *= -1.0
    scene.add(camera, pose=pose_m, name=cam.frame)
    scene.main_camera_node = next(iter(scene.get_nodes(name=cam.frame)))

    # Add Table
    table_mesh = trimesh.load_mesh(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '../'
            '../data/objects/plane/plane.obj',
        )
    )
    table_tf = RigidTransform.load(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '../',
            '../data/objects/plane/pose.tf',
        )
    )
    table_mesh.visual = table_mesh.visual.to_color()
    table_mesh.visual.vertex_colors = [[0 for c in r] for r in table_mesh.visual.vertex_colors]
    table_mesh = Mesh.from_trimesh(table_mesh)
    table_node = Node(mesh=table_mesh, matrix=table_tf.matrix)
    scene.add_node(table_node)

    # scene.add(Mesh.from_trimesh(table_mesh), pose=table_tf.matrix, name='table')
    return scene, renderer

def Get_Initial_Pose(x=0.01,y=0.01,z_lower=0.18,z_upper=0.23, rotation='SO3'):
    pose_matrix = np.eye(4)
    pose_matrix[0,3] += np.random.uniform(-x,x)
    pose_matrix[1,3] += np.random.uniform(-y,y)
    pose_matrix[2,3] += np.random.uniform(z_lower,z_upper)

    ctr_of_mass = pose_matrix[0:3, 3] 
    #TODO have some simulation where you don't rotate around center of mass

    if rotation == 'SO3':
        rand_transform = Generate_Random_TransformSO3(ctr_of_mass).dot(pose_matrix)
    elif rotation == 'z':
        rand_transform = Generate_Random_Z_Transform(ctr_of_mass).dot(pose_matrix)
    elif rotation == 'uniform':
        rand_transform = Generate_Random_Transform(ctr_of_mass).dot(pose_matrix)
    else:
        Exception()
    return rand_transform

def Sample_Mesh_Points(mesh, vertices = True, n_points = None):
    if vertices:
        points = mesh.vertices
        if n_points is not None and points.shape[0] >= n_points:
            points_clone = np.copy(points)
            np.random.shuffle(points_clone)
            points = points_clone[:n_points]
    else:
        points = mesh.sample(n_points)
        return points.T
    
def Aligned_Prism_Mesh(mesh, expand=1.1):
    mesh = mesh.copy()
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh,5, ordered=True)
    mesh.apply_transform(to_origin) # takes to oriented bb
    prism_mesh = trimesh.creation.box(extents*expand)

    obj_max, prism_max = np.max(mesh.vertices, axis = 0), np.max(prism_mesh.vertices, axis = 0)
    obj_min, prism_min = np.min(mesh.vertices, axis = 0), np.min(prism_mesh.vertices, axis = 0)
    obj_bb_center, prism_bb_center = (obj_max + obj_min) / 2, (prism_max + prism_min) / 2
    prism_center_alignment = obj_bb_center - prism_bb_center
    trans_matrix = np.eye(4)
    trans_matrix[:3,3] = prism_center_alignment
    prism_mesh.apply_transform(trans_matrix)

    prism_mesh.apply_transform(np.linalg.pinv(to_origin))
    return prism_mesh

def Load_Mesh_Path():
    config = YamlConfig(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..',
                                     'cfg/tools/data_gen_quat.yaml'))
    dataset_name_list = ['3dnet', 'thingiverse', 'kit']
    mesh_dir = config['state_space']['heap']['objects']['mesh_dir']
    mesh_dir_list = [os.path.join(mesh_dir, dataset_name) for dataset_name in dataset_name_list]
    obj_config = config['state_space']['heap']['objects']
    mesh_lists = [os.listdir(mesh_dir) for mesh_dir in mesh_dir_list]

    for mesh_dir, mesh_list in zip(mesh_dir_list, mesh_lists):
        for mesh_filename in mesh_list:
            yield mesh_dir, mesh_filename

def Load_Scale_Mesh(mesh_dir, mesh_filename, lower_scale=0.17, upper_scale=0.21):
    mesh = trimesh.load_mesh(os.path.join(mesh_dir, mesh_filename))
    random_scale = np.random.rand()
    random_scale = ((upper_scale - lower_scale) * random_scale) + lower_scale
    mesh.apply_transform(trimesh.transformations.scale_and_translate(random_scale/mesh.scale))
    return mesh

def Percent_Fit(batch, predicted_quat, expand = 1.1):
    assert predicted_quat.shape[0] == 1
    predicted_quat = predicted_quat[0]
    object_id, pose_matrix = batch['obj_id'][0], batch['pose_matrix'][0]
    true_quat = batch["quaternion"][0]
    return Wrapped_Percent_Fit(object_id, pose_matrix, predicted_quat, true_quat, expand)

def Wrapped_Percent_Fit(object_id, pose_matrix, predicted_quat, true_quat, expand = 1.1):
    obj_id = 0
    for mesh_dir, mesh_filename in Load_Mesh_Path():
        obj_id += 1
        if obj_id != object_id:
            continue
        mesh = trimesh.load_mesh(os.path.join(mesh_dir, mesh_filename))
        pose_matrix[:3,3] = 0

        prism_mesh = Aligned_Prism_Mesh(mesh, expand)
        # prism_mesh = mesh.copy()
        # prism_mesh.apply_transform(trimesh.transformations.scale_and_translate(expand))
        
        mesh.apply_transform(pose_matrix)
        prism_mesh.apply_transform(pose_matrix)

        quat = predicted_quat
        quat[0], quat[1], quat[2], quat[3] = quat[3], quat[0], quat[1], quat[2]
        mesh.apply_transform(trimesh.transformations.quaternion_matrix(quat))

        quat = true_quat
        quat[0], quat[1], quat[2], quat[3] = quat[3], quat[0], quat[1], quat[2]
        prism_mesh.apply_transform(trimesh.transformations.quaternion_matrix(quat))

        return Percent_Fit_Mesh(mesh, prism_mesh)

def Percent_Fit_Mesh(mesh, mesh_goal):
    num_sampled = 10000
    total_sampled, samples = 0, None

    # import time
    # start_time = time.time()

    while total_sampled < num_sampled:
        sampled_points = trimesh.sample.volume_mesh(mesh, (num_sampled * 6) // 5)
        total_sampled += len(sampled_points)
        samples = sampled_points if samples is None else np.concatenate((samples, sampled_points)) 

    # print("Sampled in", round(time.time() - start_time, 2), "seconds") # W/o pyembree Usually 0.5-0.7
    # start_time = time.time()

    contained_points = mesh_goal.contains(samples[:num_sampled])
    
    # print("Checked inside in", round(time.time() - start_time, 2), "seconds") # W/o pyembree Around 0.09-0.15
    # print(np.sum(contained_points), num_sampled)
    return np.sum(contained_points) / num_sampled

def negative_depth(img, ctr_of_mass):
    z = ctr_of_mass[2]
    negative = 0.8 - z - (img-(0.8-z))
    return negative

def Cut_Image(image1):
    height = image1.shape[0]
    segmask_size = np.sum(image1 <= 1 - 0.20001)
    grip = [0,0]
    while image1[grip[0]][grip[1]] > 1-0.20001:
        grip = np.random.randint(0,height,2)
    iteration, threshold = 0, 0.7
    while True:
        slope = np.random.uniform(-1,1,2)
        slope = slope[1]/np.max([slope[0], 1e-8])
        xx, yy = np.meshgrid(np.arange(0,height), np.arange(0,height))
        thiccness = height / 32 / 0.7 * threshold
        mask = (np.abs(yy-grip[1] - slope*(xx-grip[0])) <= thiccness * (np.abs(slope)+1))
        image_cut = image1.copy()
        image_cut[mask] = np.max(image1)
        # print(slope)
        if iteration % 100 == 99:
            threshold -= 0.05
        if np.sum(image_cut <= 1 - 0.20001) >= 0.7 * segmask_size:
            # print(np.sum(image_cut >= 0.200001), segmask_size)
            break
        iteration += 1
    return image_cut

def Zero_BG(image, DR = True):
    """Zeroes out all background pixels
    """
    height = image.shape[0]
    image_new = image.copy()
    mask = image_new == np.max(image_new)
    image_new[mask] = 0
    if DR:
        mask2 = np.random.randint(height // 8, (height // 8) * 7, (2,(height * height) // 160)) #100 for 128x128
        image_new[mask2[0], mask2[1]] = 0
    return image_new

def get_points_random_obj(obj_ids, points_poses, point_clouds, scales, device):
    points = [point_clouds[obj_id] / scales[obj_id] * 10 for obj_id in obj_ids]
    # print(batch["pose_matrix"][0])
    points, points_poses = torch.Tensor(points).to(device), torch.Tensor(points_poses).to(device)
    points = torch.bmm(points_poses, points)
    # print(points[:,:5])
    return points

def get_points_numpy(obj_ids, points_poses, point_clouds, scales, device):
    points = point_clouds[obj_ids-1]
    # print(batch["pose_matrix"][0])
    points, points_poses = torch.Tensor(points).to(device), torch.Tensor(points_poses).to(device)
    points = torch.bmm(points_poses, points)
    # print(points[:,:5])
    return points

def get_points_single_obj(obj_ids, points_poses, point_clouds, scales, device):
    assert np.sum(obj_ids) == obj_ids[0] * len(obj_ids), "Obj_ID: {}".format(obj_ids)
    pc1 = point_clouds[obj_ids[0]] / scales[obj_ids[0]] * 10
    points = np.tile(pc1, (points_poses.shape[0],1,1))
    # print(batch["pose_matrix"][0])
    points, points_poses = torch.Tensor(points).to(device), torch.Tensor(points_poses).to(device)
    points = torch.bmm(points_poses, points)
    # print(points[:,:5])
    return points

def Quantize(img, demean=False):
    if demean:
        img_max = img.max()
        img_min = img[img!=0].min()
        img_range = img_max - img_min
        img[img!=0] = (img[img!=0] - img_min + 0.00001) / (img_range*2 + 0.00001) + 0.5
    return (img * 65535).astype(int) / 65535

def get_points_vertices(obj_ids, points_poses, point_clouds, scales, device):
    """obj_ids: (batch,)
    points_poses: (batch, 3, 3)
    point_clouds: dict of (3, ?)
    scales: dict of ints
    """
    pc1, pc2 = point_clouds[obj_ids[0]]/scales[obj_ids[0]]*10, point_clouds[obj_ids[-1]]/scales[obj_ids[-1]]*10
    n1, n2 = pc1.shape[1], pc2.shape[1]

    if obj_ids[0] != obj_ids[-1]:
        # print("Changing objects this batch!")
        if n1 != n2:
            num_points = min((n1,n2))
            indices1, indices2 = np.random.choice(n1, num_points, replace=False), np.random.choice(n2, num_points, replace=False)
            pc1, pc2 = pc1.T[indices1].T, pc2.T[indices2].T
        b1, b2 = np.sum(obj_ids == obj_ids[0]), np.sum(obj_ids == obj_ids[-1])
        points1, points2 = np.tile(pc1, (b1,1,1)), np.tile(pc2, (b2,1,1))
        # print(points1.shape, points2.shape)
        points = np.concatenate((points1, points2))
    else:
        points = np.tile(pc1, (obj_ids.shape[0],1,1))
    
    points, points_poses = torch.Tensor(points).to(device), torch.Tensor(points_poses).to(device)
    points = torch.bmm(points_poses, points)
    return points
