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
from .plot_utils import *
from .utils import *

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

def Axis_String(axis):
    axis = np.round(axis, 3)
    return "[" +str(axis[0]) + ", " + str(axis[1]) + ", " + str(axis[2]) + "]"

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

def Quat_to_Lie(q):
    omega, _ = cv2.Rodrigues(Quaternion_to_Rotation(q, np.zeros(3))[:3,:3])    
    return omega.flatten()

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

def error2angle(err):
    return np.arccos(1-err) * 180 / np.pi * 2

def normalize(z):
    return z / np.linalg.norm(z)
