import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation
import torch
import torchvision
from autolab_core import YamlConfig, RigidTransform

def normalize(z):
    return z / np.linalg.norm(z)

def Generate_Quaternion(start=0, end=np.pi/6):
    """Generate a random quaternion with conditions.
    To avoid double coverage and limit our rotation space, 
    we make sure the real component is positive and have 
    the greatest magnitude. Sample axes randomly. Sample degree uniformly
    """
    axis = np.random.normal(0, 1, 3)
    axis = axis / np.linalg.norm(axis) 
    angle = np.random.uniform(start,end)
    quat = Rotation.from_rotvec(axis * angle).as_quat()
    if quat[3] < 0:
        quat = -1 * quat
    # print("Quaternion is ", 180/np.pi*np.linalg.norm(Rotation.from_quat(random_quat).as_rotvec()))
    return quat

def Generate_Quaternion_SO3():
    """Generate a random quaternion with conditions.
    To avoid double coverage and limit our rotation space, 
    we make sure the real component is positive and have 
    the greatest magnitude. We also limit rotations to less
    than 60 degrees. We sample according to the following links:
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
    http://planning.cs.uiuc.edu/node198.html
    """
    quat = np.zeros(4)
    # while np.max(np.abs(quat)) < 0.866: # 60 degrees
    # while np.max(np.abs(quat)) < 0.92388: # 45 degrees
    while np.max(np.abs(quat)) < 0.96592:  # 30 degrees
      uniforms = np.random.uniform(0, 1, 3)
      one_minus_u1, u1 = np.sqrt(1 - uniforms[0]), np.sqrt(uniforms[0])
      uniforms_pi = 2*np.pi*uniforms
      quat = np.array(
          [one_minus_u1 * np.sin(uniforms_pi[1]),
           one_minus_u1 * np.cos(uniforms_pi[1]),
           u1 * np.sin(uniforms_pi[2]),
           u1 * np.cos(uniforms_pi[2])])

    max_i = np.argmax(np.abs(quat))
    quat[3], quat[max_i] = quat[max_i], quat[3]
    if quat[3] < 0:
        quat = -1 * quat
    # print("Quaternion is ", 180/np.pi*np.linalg.norm(Rotation.from_quat(random_quat).as_rotvec()))
    return quat

def Quaternion_String(quat):
    """Converts a 4 element quaternion to a string for printing
    """
    quat = np.round(quat, 3)
    return str(quat[3]) + " + " + str(quat[0]) + "i + " + str(quat[1]) + "j + " + str(quat[2]) + "k"

def Quaternion_to_Rotation(quaternion, center_of_mass):
    """Take in an object's center of mass and a quaternion, and
    return a rotation matrix.
    """
    rotation_vector = Rotation.from_quat(quaternion).as_rotvec()
    angle = np.linalg.norm(rotation_vector)
    axis = rotation_vector / angle
    return RigidTransform.rotation_from_axis_and_origin(axis=axis, origin=center_of_mass, angle=angle).matrix

def Rotation_to_Quaternion(rot_matrix):
    """Take in an object's 4x4 pose matrix and return a quaternion
    """
    quat = Rotation.from_dcm(rot_matrix[:3,:3]).as_quat()
    if quat[3] < 0:
        quat = -quat
    return quat

def Generate_Random_TransformSO3(center_of_mass):
    """Create a matrix that will randomly rotate an object about an axis by a randomly sampled quaternion
    """
    quat = np.zeros(4)
    uniforms = np.random.uniform(0, 1, 3)
    one_minus_u1, u1 = np.sqrt(1 - uniforms[0]), np.sqrt(uniforms[0])
    uniforms_pi = 2*np.pi*uniforms
    quat = np.array(
        [one_minus_u1 * np.sin(uniforms_pi[1]),
        one_minus_u1 * np.cos(uniforms_pi[1]),
        u1 * np.sin(uniforms_pi[2]),
        u1 * np.cos(uniforms_pi[2])])

    quat = normalize(quat)
    return Quaternion_to_Rotation(quat, center_of_mass)

def Generate_Random_Transform(center_of_mass):
    """Create a matrix that will randomly rotate an object about an axis by a random angle between 0 and 45.
    """
    angle = 1/4*np.pi*np.random.random()
    # print(angle * 180 / np.pi)
    while True:
        axis = np.random.normal(0,1,3)
        if np.linalg.norm(axis) < 1:
            axis = axis / np.linalg.norm(axis)
            break
    return RigidTransform.rotation_from_axis_and_origin(axis=axis, origin=center_of_mass, angle=angle).matrix

def Generate_Random_Z_Transform(center_of_mass):
    """Create a matrix that will randomly rotate an object about the z-axis by a random angle.
    """
    z_angle = 2*np.pi*np.random.random()
    return RigidTransform.rotation_from_axis_and_origin(axis=[0, 0, 1], origin=center_of_mass, angle=z_angle).matrix

def Plot_Datapoint(datapoint):
    """Takes in a datapoint of our Tensor Dataset, and plots its two images for visualizing their 
    iniitial pose and rotation.
    """
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    fig1 = plt.imshow(datapoint["depth_image1"][:, :, 0], cmap='gray', vmin = 0.7)
    plt.title('Stable pose')
    plt.subplot(122)
    fig2 = plt.imshow(datapoint["depth_image2"][:, :, 0], cmap='gray')
    fig1.axes.get_xaxis().set_visible(False)
    fig1.axes.get_yaxis().set_visible(False)
    fig2.axes.get_xaxis().set_visible(False)
    fig2.axes.get_yaxis().set_visible(False)
    plt.title('After Rigid Transformation: ' + Quaternion_String(datapoint["quaternion"]))
    plt.show()
    # plt.savefig("pictures/allobj/obj" + str(datapoint['obj_id']) + ".png")
    # plt.close()

def error2angle(err):
    return np.arccos(1-err) * 180 / np.pi * 2

def Plot_Loss(loss_history, loss_plot_fname):
    """Plots the training and validation loss, provided that there is a config file with correct
    location of data
    """
    losses = pickle.load(open(loss_history, "rb"))
    train_returns = np.array(losses["train_loss"])
    test_returns = np.array(losses["test_loss"])

    # if config['loss'] == 'cosine':
    #     train_returns = np.arccos(1-train_returns) * 180 / np.pi * 2
    #     test_returns = np.arccos(1-test_returns) * 180 / np.pi * 2
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(train_returns)) + 1, train_returns, label="Training Loss")
    plt.plot(np.arange(len(test_returns)) + 1, test_returns, label="Testing Loss")
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend(loc='best')
    plt.savefig(loss_plot_fname)
    plt.close()

def Plot_Angle_vs_Loss(quaternions, losses, mean_loss, fname, max_angle = 30):
    bins = max_angle // 5
    rotation_angles = [[] for i in range(bins)]
    for q,l in zip(quaternions, losses):
        rot_vec = Rotation.from_quat(q).as_rotvec()
        rot_angle = np.linalg.norm(rot_vec) * 180 / np.pi
        bin_num = np.min((int(rot_angle // 5), bins-1))
        rotation_angles[bin_num].append(l)

    mean_losses = [error2angle(np.mean(ra)) for ra in rotation_angles]
    errors = [error2angle(np.std(ra)) for ra in rotation_angles]
    labels = [str(i) + "-" + str(i+5) for i in range(0,max_angle,5)]

    plt.figure(figsize=(10,5))
    plt.bar(labels, mean_losses, yerr = errors)
    plt.axhline(mean_loss, c = 'r')
    plt.xlabel("Rotation Angle (Degrees)")
    plt.ylabel("Angle Loss (Degrees)")
    plt.ylim(0.0, (np.max(mean_losses)+np.max(errors))*1.1)
    plt.title("Loss vs Rotation Angle")
    plt.savefig(fname)
    plt.close()

def Plot_Small_Angle_Loss(quaternions, losses, mean_loss):
    bins = 10
    rotation_angles = [[] for i in range(bins)]
    for q,l in zip(quaternions, losses):
        rot_vec = Rotation.from_quat(q).as_rotvec()
        rot_angle = np.linalg.norm(rot_vec) * 180 / np.pi
        if rot_angle <= 10:
            bin_num = np.min((int(rot_angle), bins-1))
            rotation_angles[bin_num].append(l)

    mean_losses = [error2angle(np.mean(ra)) for ra in rotation_angles]
    errors = [error2angle(np.std(ra)) for ra in rotation_angles]
    labels = [str(i) + "-" + str(i+1) for i in range(0,bins,1)]

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

def Quantize(images):
    return (images * 65535).astype(int)

def addNoise(image, std=0.001):
    """Adds noise to image array.
    """
    noise = np.random.normal(0, std, image.shape)
    return image + noise

def display_conv_layers(model):
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    with torch.no_grad():
        imshow(torchvision.utils.make_grid(model.resnet.conv1.weight))
