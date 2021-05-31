import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation
import torch
import torchvision
from autolab_core import YamlConfig, RigidTransform
from pyquaternion import Quaternion
import cv2
from mpl_toolkits.mplot3d import Axes3D
from perception import CameraIntrinsics, RgbdSensorFactory, Image, DepthImage
import os
import trimesh
from .utils import *
from .rotation_utils import *

def Plot_Bad_Predictions(dataset, predicted_quats, true_quaternions, angle_vs_losses, test_indices, name = "worst"):
    """Takes in the dataset, predicted quaternions, and losses of the 
    worst predictions in the validation set
    """
    losses = []
    for obj_id, angle,cosine_loss,sm_loss,fit_loss in angle_vs_losses:
        losses.append(sm_loss)

    if name == "worst":
        indices = np.argsort(losses)[-5:-1]
    elif name == "best":
        smallest_losses_idx = np.argsort(losses)
        smallest_losses = []
        for i in smallest_losses_idx:
            if true_quaternions[i][3] < 0.975:
                smallest_losses.append(i)
            if len(smallest_losses) >= 5:
                break
        indices = np.array(smallest_losses)
    
    for i in indices:
        datapoint = dataset.get_item_list(test_indices[i:i+1])
        predicted_quat, image1, image2 = predicted_quats[i], datapoint["depth_image1"][0][0], datapoint["depth_image2"][0][0]
        plt.rc('text', usetex=True)
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.axis('off')
        img_range = np.max(image1) - np.min(image1[image1 != 0]) + 0.0001
        plt.imshow(image1, cmap='gray', vmin = np.min(image1[image1 != 0]) - img_range * 0.1)
        plt.title(r"$I^s$")

        plt.subplot(132)
        plt.axis('off')
        img_range = np.max(image2) - np.min(image2[image2 != 0]) + 0.0001
        plt.imshow(image2, cmap='gray', vmin = np.min(image2[image2 != 0]) - img_range * 0.1)
        plt.title('True Quat: ' + Quaternion_String(datapoint["quaternion"][0]))

        plt.subplot(133)
        plt.axis('off')
        image3 = Plot_Predicted_Rotation(datapoint, predicted_quat)
        img_range = np.max(image3) - np.min(image3[image3 != 0]) + 0.0001
        plt.imshow(image3, cmap='gray', vmin = np.min(image3[image3 != 0]) - img_range * 0.1)
        plt.title('Pred Quat: ' + Quaternion_String(predicted_quat))

        plt.tight_layout(pad=2)
        plt.savefig("plots/worst_preds/" + name + "_pred_" + 
                        str(datapoint['obj_id'][0]) + "_" + str(losses[i])[2:5])
        print(losses[i])
        # plt.savefig("plots/worst_preds/" + name + "_pred_" + str(datapoint['obj_id'][0]) + "_"
        # + str(1- np.dot(predicted_quat, datapoint['quaternion'].flatten()))[2:5])
        # print(1 - np.dot(predicted_quat, datapoint['quaternion'].flatten()))
        # plt.show()
        plt.close()

def Plot_Predicted_Rotation(datapoint, predicted_quat):
    object_id, pose_matrix = datapoint['obj_id'], datapoint['pose_matrix'][0]
    scene, renderer = create_scene_real()
    obj_id = 0
    for mesh_dir, mesh_filename in Load_Mesh_Path():
        obj_id += 1
        if obj_id != object_id:
            continue

        # load object mesh
        mesh = Load_Scale_Mesh(mesh_dir, mesh_filename)

        obj_mesh = Mesh.from_trimesh(mesh)
        object_node = Node(mesh=obj_mesh, matrix=np.eye(4))

        scene.add_node(object_node)
        # print(pose_matrix)
        ctr_of_mass = pose_matrix[0:3, 3]

        new_pose = Quaternion_to_Rotation(predicted_quat, ctr_of_mass).dot(pose_matrix)
        scene.set_pose(object_node, pose=new_pose)

        # new_pose2 = Quaternion_to_Rotation(datapoint['quaternion'][0], ctr_of_mass) @ pose_matrix
        # scene.set_pose(object_node2, pose=new_pose2)

        image = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
        return Zero_BG(Crop_Image(image), DR = False)

def Plot_Predicted_Rotation_Prism(datapoint, predicted_quat):
    object_id, pose_matrix = datapoint['obj_id'], datapoint['pose_matrix'][0]
    scene, renderer = create_scene()
    obj_id = 0
    for mesh_dir, mesh_filename in Load_Mesh_Path():
        obj_id += 1
        if obj_id != object_id:
            continue

        mesh = Load_Scale_Mesh(mesh_dir, mesh_filename)
        obj_mesh = Mesh.from_trimesh(mesh)
        object_node = Node(mesh=obj_mesh, matrix=np.eye(4))

        prism_mesh = Aligned_Prism_Mesh(mesh)

        prism_obj_mesh = Mesh.from_trimesh(prism_mesh)
        prism_node = Node(mesh=prism_obj_mesh, matrix=np.eye(4))

        scene.add_node(object_node) 
        ctr_of_mass = pose_matrix[0:3, 3]

        new_pose = Quaternion_to_Rotation(predicted_quat, ctr_of_mass).dot(pose_matrix)
        scene.set_pose(object_node, pose=new_pose)

        gt_pose = Quaternion_to_Rotation(datapoint['quaternion'][0], ctr_of_mass).dot(pose_matrix)
        scene.set_pose(prism_node, pose=gt_pose)

        image = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
        return Zero_BG(image, DR = False)

def Plot_PC(pointclouds, filename = "plots/test.png"):
    fig = plt.figure()
    ax = Axes3D(fig)
    for pc in pointclouds:
        ax.scatter(pc[:,0], pc[:,1], pc[:,2])
    ax.view_init(elev=90, azim=270) #topdown like camera
    # ax.view_init(elev=45, azim=270)
    # plt.show()
    plt.savefig(filename)
    plt.close()

def Plot_Image(img, fname="test.png"):
    img_range = np.max(img) - np.min(img[img != 0]) + 0.0001
    plt.imshow(img, cmap='gray', vmin = np.min(img[img != 0]) - img_range * 0.1)
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname)
    plt.close()    

def Plot_Datapoint(image1, image2, quat, zeroed=True, show = False):
    """Takes in a datapoint of our Tensor Dataset, and plots its two images for visualizing their 
    initial pose and rotation.
    """
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.axis('off')
    plt.title(r"$I^s$")
    if zeroed:
        img_range = np.max(image1) - np.min(image1[image1 != 0]) + 0.0001
        plt.imshow(image1, cmap='gray', vmin = np.min(image1[image1 != 0]) - img_range * 0.1)
        plt.subplot(122)
        img_range = np.max(image2) - np.min(image2[image2 != 0]) + 0.0001
        plt.imshow(image2, cmap='gray', vmin = np.min(image2[image2 != 0]) - img_range * 0.1)
    else:
        mask = np.where(image1 < 0.7999)
        plt.imshow(image1, cmap='gray', vmax = np.max(image1[mask[0], mask[1]]) + 0.02)
        plt.subplot(122)
        mask = np.where(image2 < 0.7999)
        plt.imshow(image2, cmap='gray', vmax = np.max(image2[mask[0], mask[1]]) + 0.02)
    plt.axis("off")
    
    rotation_vector = Rotation.from_quat(quat).as_rotvec()
    angle_radians = np.linalg.norm(rotation_vector)
    axis = rotation_vector / angle_radians
    angle = round(angle_radians * 180 / np.pi, 3)
    plt.title('After Quaternion Rotation: ' + Quaternion_String(quat) + 
                    "\n After rotating: {} degrees around ".format(angle) + Axis_String(axis))
    plt.tight_layout(pad=2)
    if show:
        plt.show()
    else:
        plt.savefig("plots/test_datapoint.png")
        plt.close()
    # plt.savefig("pictures/allobj/obj" + str(datapoint['obj_id']) + ".png")

def Plot_Loss(loss_history, loss_plot_fname):
    """Plots the training and validation loss, provided that there is a config file with correct
    location of data
    """
    losses = pickle.load(open(loss_history, "rb"))
    train_returns = np.array(losses["train_loss"])
    test_returns = np.array(losses["test_loss"])
    min_train = np.round(np.min(train_returns), 3)
    min_test = np.round(np.min(test_returns), 3)
    # if config['loss'] == 'cosine':
    #     train_returns = np.arccos(1-train_returns) * 180 / np.pi * 2
    #     test_returns = np.arccos(1-test_returns) * 180 / np.pi * 2
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(train_returns)) + 1, train_returns, label="Training Loss, min: {}".format(min_train))
    plt.plot(np.arange(len(test_returns)) + 1, test_returns, label="Testing Loss, min: {}".format(min_test))
    plt.ylim(min_train, np.max(train_returns))
    # plt.ylim(min_train, 0.14)
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend(loc='best')
    plt.savefig(loss_plot_fname)
    plt.close()

def Plot_Angle_vs_Loss(angle_vs_losses, fname, loss="shapematch", max_angle=30):
    bins = max_angle // 5
    rotation_angles = [[] for i in range(bins)]
    mean_loss = 0
    for obj_id,angle,l,sm,fit in angle_vs_losses:
        bin_num = np.min((int(angle // 5), bins-1))
        if loss == "cosine":
            rotation_angles[bin_num].append(l)
            mean_loss += l
        elif loss == "shapematch":
            rotation_angles[bin_num].append(sm)
            mean_loss += sm
        elif loss == "fit":
            rotation_angles[bin_num].append(1-fit)
            mean_loss += 1-fit

    mean_loss = mean_loss / len(angle_vs_losses)

    if loss == "cosine":
        mean_losses = [error2angle(np.mean(ra)) for ra in rotation_angles]
        errors = [error2angle(np.std(ra)) for ra in rotation_angles]
        labels = [str(i) + "-" + str(i+5) for i in range(0,max_angle,5)]
        mean_loss = error2angle(mean_loss)
    else:
        mean_losses = [np.mean(ra) for ra in rotation_angles]
        errors = [np.std(ra) for ra in rotation_angles]
        labels = [str(i) + "-" + str(i+5) for i in range(0,max_angle,5)]

    plt.figure(figsize=(10,5))
    plt.bar(labels, mean_losses, yerr = errors)
    y_max = (np.max(mean_losses) + np.max(errors))*1.1
    ax = plt.gca()
    for i, v in enumerate(np.round(mean_losses,3)):
        plt.text(i-0.175,y_max/30,str(v), color='cyan', va='center', fontweight='bold')
    plt.axhline(mean_loss, c='r', label = "Mean overall: {}".format(round(mean_loss,3)))
    plt.ylim(0.0, y_max)
    plt.xlabel("Rotation Angle (Degrees)", fontsize=20)
    # print("Plot_Loss Mean is:", mean_loss)
    if loss == "cosine":
        plt.title("Angle vs Rotation Angle", fontsize=20)
        plt.ylabel("Angle Loss (Degrees)", fontsize=20)
    elif loss == "shapematch":
        plt.title("Shape-Match Loss vs Rotation Angle", fontsize=20)
        plt.ylabel("Shape-Match Loss", fontsize=20)
    elif loss == "fit":
        plt.title("Percent Fit vs Rotation Angle", fontsize=20)
        plt.ylabel("Mean Percent Fit", fontsize=20)
        plt.axhline(0.773, c='lime', label = "Random quaternion baseline: {}".format(0.773)) #6638 surface metric
    plt.legend()

    plt.tight_layout(pad=1)
    plt.savefig(fname)
    plt.close()

def Plot_Eccentricity_vs_Fit(angle_vs_losses, fname, angle_lower):
    ecc = pickle.load(open("cfg/tools/data/eccentricities", "rb"))
    for obj_id in ecc.keys():
        ecc[obj_id] -= 1
    # thresholds = [0.0,0.1,0.3,0.6,1.0,1.5,2.5,4.0,6.0,9.0,100.0]
    thresholds = [2.0,2.5,3.0,4.0,5.0,6.0,7.5,9.0,100.0]
    assert angle_lower == 0 or angle_lower == 10 or angle_lower == 20
    baseline_fit = [0.80, 0.78, 0.74][angle_lower // 10] #surface was [0.687, 0.670, 0.636]
    bins = len(thresholds) - 1
    ecc_lst = [[] for i in range(bins)]
    total_loss, n_points = 0,0
    for obj_id,angle,l,sm,fit in angle_vs_losses:
        if angle >= angle_lower and angle < angle_lower + 10:
            bin_num = -1
            for thresh in thresholds:
                if ecc[obj_id] > thresh:
                    bin_num += 1
            ecc_lst[bin_num].append(1-fit)
            total_loss += 1-fit
            n_points += 1

    mean_loss = total_loss / n_points
    mean_losses = [np.mean(ecc_bin) for ecc_bin in ecc_lst]
    errors = [np.std(ecc_bin) for ecc_bin in ecc_lst]
    labels = [str(thresholds[i]) + "-" + str(thresholds[i+1]) for i in range(bins)]
    print(mean_loss)
    print(mean_losses)
    print(errors)
    plt.figure(figsize=(10,5))
    plt.bar(labels, mean_losses, yerr = errors)
    y_max = (np.max(mean_losses) + np.max(errors))*1.1
    ax = plt.gca()
    for i, v in enumerate(np.round(mean_losses,3)):
        plt.text(i-0.25,y_max/30,str(v), color='cyan', va='center', fontweight='bold')
    plt.axhline(mean_loss, c='r', label = "Mean overall: {}".format(round(mean_loss,3)))
    plt.ylim(0.0, y_max)
    plt.title("Percent Fit vs Eccentricity, {}-{} Degree Rotations".format(angle_lower,angle_lower+10), fontsize=20)
    plt.xlabel("Eccentricity", fontsize=20)
    # print("Plot_Loss Mean is:", mean_loss)
    plt.ylabel("Mean Percent Fit", fontsize=20)
    plt.axhline(baseline_fit, c='lime', label = "Random quaternion baseline: {}".format(baseline_fit))
    plt.legend()

    plt.tight_layout(pad=1)
    plt.savefig(fname + "_ecc_" + str(angle_lower))
    plt.close()

def Plot_Small_Angle_Loss(angle_vs_losses, loss = "shapematch"):
    bins, mean_loss = 10, 0
    rotation_angles = [[] for i in range(bins)]
    for angle,l,sm in angle_vs_losses:
        if angle <= 10:
            bin_num = np.min((int(angle), bins-1))
            if loss == "cosine":
                rotation_angles[bin_num].append(l)
                mean_loss += l
            else:
                rotation_angles[bin_num].append(sm)
                mean_loss += sm
    mean_loss = mean_loss / len(angle_vs_losses)

    if loss == "cosine":
        mean_losses = [error2angle(np.mean(ra)) for ra in rotation_angles]
        errors = [error2angle(np.std(ra)) for ra in rotation_angles]
        labels = [str(i) + "-" + str(i+1) for i in range(0, 10)]
    else:
        mean_losses = [np.mean(ra) for ra in rotation_angles]
        errors = [np.std(ra) for ra in rotation_angles]
        labels = [str(i) + "-" + str(i+1) for i in range(0, 10)]

    plt.figure(figsize=(10,5))
    plt.bar(labels, mean_losses, yerr = errors)
    plt.axhline(mean_loss, c = 'r')
    plt.xlabel("Rotation Angle (Degrees)")
    plt.ylabel("Angle Loss (Degrees)")
    plt.ylim(0.0, (np.max(mean_losses)+np.max(errors))*1.1)
    plt.title("Loss vs Rotation Angle")
    plt.savefig("plots/small_rot.png")
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
    mean_losses = [error2angle(np.mean(ra)) for ra in rotation_angles]
    errors = [error2angle(np.std(ra)) for ra in rotation_angles]
    plt.figure(figsize=(10,5))
    plt.bar(labels, mean_losses, yerr = errors)
    plt.axhline(mean_loss, c = 'r')
    plt.xlabel("Rotation Angle from Z-Axis (Degrees)")
    plt.ylabel("Angle Loss (Degrees)")
    plt.ylim(0.0, (np.max(mean_losses)+np.max(errors))*1.1)
    plt.title("Loss vs Rotation Angle from Z-Axis")
    plt.savefig("plots/axes_loss.png")
    plt.close()
