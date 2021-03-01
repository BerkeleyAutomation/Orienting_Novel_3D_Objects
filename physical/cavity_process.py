import matplotlib
matplotlib.use('Agg')
from autolab_core import YamlConfig, RigidTransform, TensorDataset, Box, Logger, PointCloud
import os
import time
import sys
import numpy as np
from tools.utils import *
from phoxipy import PhoXiSensor, ColorizedPhoXiSensor
from perception import BinaryImage

def Plot_Image(image, filename):
    """x
    """
    # cv2.imwrite(filename, image)
    img_range = np.max(image) - np.min(image[image != 0]) + 0.0001
    plt.imshow(image, cmap='gray', vmin = np.min(image[image != 0]) - img_range * 0.1, vmax = np.max(image))
    plt.axis('off')
    plt.savefig(filename)
    # plt.show()
    plt.close()

class Cavity:
    def __init__(self, sensor=None, plot=False):
        calib_dir = '/nfs/diskstation/calib/phoxi'
        phoxi_config = YamlConfig("physical/cfg/tools/colorized_phoxi.yaml")
        self.sensor_name = 'phoxi'
        self.sensor_config = phoxi_config['sensors'][self.sensor_name]

        self.sensor_type = self.sensor_config['type']
        self.sensor_frame = self.sensor_config['frame']
        if sensor:
            self.sensor = sensor
        else:
            self.sensor = ColorizedPhoXiSensor("1703005", 3, calib_dir='/nfs/diskstation/calib/', inpaint=False) 
            self.sensor.start()

        # logger.info('Ready to capture images from sensor %s' %(self.sensor_name))
        # basedir = "/home/shivin/catkin_ws/src/ambidex/tests/cfg/"
        # yaml_obj_loader = YamlObjLoader(basedir)

        # self.robot = yaml_obj_loader('physical_yumi_no_jaw')
        # self.home_pose = self.robot.right.get_pose()
        self.camera_intr = CameraIntrinsics.load(os.path.join(calib_dir, 'phoxi.intr'))
        self.T_camera_world = RigidTransform.load(os.path.join(calib_dir, 'phoxi_to_world.tf'))

        # x,y,z = self.home_pose.translation
        # self.workspace = Box(np.array([x-0.1, y-0.075,z-0.05]),
        #                     np.array([x+0.1,y+0.075, z+0.15]), frame = "world")
        self.plane_workspace = Box(np.array(phoxi_config['plane_workspace']['min_pt']),
                            np.array(phoxi_config['plane_workspace']['max_pt']),
                            frame='world')

        self.box_workspace = Box(np.array(phoxi_config['box_workspace']['min_pt']),
                            np.array(phoxi_config['box_workspace']['max_pt']),
                            frame='world')

        self.negative_workspace = Box(np.array(phoxi_config['negative_workspace']['min_pt']),
                            np.array(phoxi_config['negative_workspace']['max_pt']),
                            frame='world')

        self.base_path = "physical/cavities/"
        self.plot = plot

    def capture_image_ideal(self, center_i=None, center_j=None, frame=0):
        # logger.info('Capturing depth image' + str(frame))
        img = self.sensor.read()
        rgb_1, depth_1 = img.color, img.depth

        depth_2, ideal_T_world_phoxi = self.top_down_img(depth_1)

        # deproject into 3D world coordinates
        point_cloud_cam = self.camera_intr.deproject(depth_2)
        point_cloud_cam.remove_zero_points()
        point_cloud_world = ideal_T_world_phoxi * point_cloud_cam
        seg_point_cloud_world, _ = point_cloud_world.box_mask(self.plane_workspace)
        
        # compute the segmask for points above the box
        seg_point_cloud_cam = ideal_T_world_phoxi.inverse() * seg_point_cloud_world
        depth_3 = self.camera_intr.project_to_image(seg_point_cloud_cam)
        segmask = depth_3.to_binary()

        z_thresh = np.mean(seg_point_cloud_world.data,1)[2]
        seg2_point_cloud_world = seg_point_cloud_world[seg_point_cloud_world.data[2] < z_thresh]
        seg2_point_cloud_cam = ideal_T_world_phoxi.inverse() * seg2_point_cloud_world
        depth_4 = self.camera_intr.project_to_image(seg2_point_cloud_cam)
        # import pdb
        # pdb.set_trace()
        z_max = seg2_point_cloud_world.data.max(1)[2]
        seg2_point_cloud_world._data[2,:] = z_max - seg2_point_cloud_world.data[2,:] + z_max + 0.25
        seg2_point_cloud_cam = ideal_T_world_phoxi.inverse() * seg2_point_cloud_world
        depth_5 = self.camera_intr.project_to_image(seg2_point_cloud_cam)
        depth_6 = self.negate_depth(depth_4)
        plot_depth(depth_5)

        # ind = np.where(depth_3.data != 0)
        # if not center_i or not center_j:
        #     center_i, center_j = np.mean(ind[0]), np.mean(ind[1])
        # height, width = np.max(ind[0])- np.min(ind[0]), np.max(ind[1])- np.min(ind[1])
        # print(height, width, center_i, center_j)

        # depth_int = depth_3.crop(256, 256, center_i, center_j)
        # depth_4 = depth_int.resize((128,128), 'nearest')
        # depth_seg4.save('ros_phoxi/depth_seg4_%d.png' %(frame))
        # I_sg = depth_4.data
        # mask = I_sg != 0
        # I_sg[mask] = I_sg[mask] - np.min(depth_seg4.data)
        # I_sg[mask] = (I_sg[mask] / np.max(I_sg[mask]) * 0.1) + 0.5
        frame_string = str(frame).zfill(2)
        Plot_Image(rgb_1.data, self.base_path + "/rgb_1" + frame_string + ".png")
        # Plot_Image(rgb_2.data, self.base_path + "/rgb_2" + frame_string + ".png")
        # Plot_Image(rgb_3.data, self.base_path + "/rgb_3" + frame_string + ".png")
        Plot_Image(depth_1.data, self.base_path + "/depth_1" + frame_string + ".png")
        Plot_Image(depth_2.data, self.base_path + "/depth_2" + frame_string + ".png")
        Plot_Image(depth_3.data, self.base_path + "/depth_3" + frame_string + ".png")
        Plot_Image(depth_4.data, self.base_path + "/depth_4" + frame_string + ".png")
        Plot_Image(depth_5.data, self.base_path + "/depth_5" + frame_string + ".png")
        Plot_Image(depth_6.data, self.base_path + "/depth_6" + frame_string + ".png")

        # return I_sg, center_i, center_j
        return depth_5

    def capture_image(self, center_i=None, center_j=None, frame=0):
        # logger.info('Capturing depth image' + str(frame))
        img = self.sensor.read()
        rgb_1, depth_1 = img.color, img.depth

        # deproject into 3D world coordinates
        point_cloud_cam = self.camera_intr.deproject(depth_1)
        point_cloud_cam.remove_zero_points()
        point_cloud_world = self.T_camera_world * point_cloud_cam
        seg_point_cloud_world, _ = point_cloud_world.box_mask(self.plane_workspace)
        
        # compute the segmask for points above the box
        seg_point_cloud_cam = self.T_camera_world.inverse() * seg_point_cloud_world
        depth_2 = self.camera_intr.project_to_image(seg_point_cloud_cam)
        segmask = depth_2.to_binary()
        rgb_2 = rgb_1.mask_binary(segmask)

        seg2_point_cloud_world = seg_point_cloud_world[seg_point_cloud_world.data[2]>seg_point_cloud_world.data.mean(1)[2]]
        seg2_point_cloud_cam = self.T_camera_world.inverse() * seg2_point_cloud_world
        depth_3 = self.camera_intr.project_to_image(seg2_point_cloud_cam)
        segmask = depth_3.to_binary()
        rgb_3 = rgb_2.mask_binary(segmask)

        # ind = np.where(depth_3.data != 0)
        # if not center_i or not center_j:
        #     center_i, center_j = np.mean(ind[0]), np.mean(ind[1])
        # height, width = np.max(ind[0])- np.min(ind[0]), np.max(ind[1])- np.min(ind[1])
        # print(height, width, center_i, center_j)

        # depth_int = depth_3.crop(256, 256, center_i, center_j)
        # depth_4 = depth_int.resize((128,128), 'nearest')
        # depth_seg4.save('ros_phoxi/depth_seg4_%d.png' %(frame))
        # I_sg = depth_4.data
        # mask = I_sg != 0
        # I_sg[mask] = I_sg[mask] - np.min(depth_seg4.data)
        # I_sg[mask] = (I_sg[mask] / np.max(I_sg[mask]) * 0.1) + 0.5

        frame_string = str(frame).zfill(2)
        Plot_Image(rgb_1.data, self.base_path + "/rgb_1" + frame_string + ".png")
        Plot_Image(rgb_2.data, self.base_path + "/rgb_2" + frame_string + ".png")
        Plot_Image(rgb_3.data, self.base_path + "/rgb_3" + frame_string + ".png")
        Plot_Image(depth_1.data, self.base_path + "/depth_1" + frame_string + ".png")
        Plot_Image(depth_2.data, self.base_path + "/depth_2" + frame_string + ".png")
        Plot_Image(depth_3.data, self.base_path + "/depth_3" + frame_string + ".png")
        # Plot_Image(depth_4.data, self.base_path + "/depth_4" + frame_string + ".png")

        # return I_sg, center_i, center_j

    def top_down_img(self, depth):
        height_slope = 10/700
        width_slope = 100/850
        # plot_depth(depth)
        pc = self.camera_intr.deproject(depth)
        pc = self.T_camera_world * pc
        ideal_T_world_phoxi = self.T_camera_world.copy()
        ideal_T_world_phoxi.rotation = np.array([[0.0, -1.0,  0.0],
                                                [-1.0, 0.0,  0.0],
                                                [0.0,  0.0, -1.0]])
        ideal_T_world_phoxi.translation[:2] = np.mean(pc.data,1)[:2]
        ideal_T_world_phoxi.translation[2] = 0.8
        point_cloud_camera = ideal_T_world_phoxi.inverse() * pc
        depth = self.camera_intr.project_to_image(point_cloud_camera)
        # plot_depth(depth)
        return depth, ideal_T_world_phoxi

    def negate_depth(self, depth):
        plot_depth(depth) if self.plot else 0
        img_min = depth.data[depth.data != 0].min()
        depth = depth.copy()
        depth._data[depth.data != 0] = (img_min - depth.data[depth.data != 0] + img_min)[:,None]
        plot_depth(depth) if self.plot else 0
        return depth

    def process_clamshell_positive(self):
        img = self.sensor.read()
        rgb_1, depth_1 = img.color, img.depth

        seg_point_cloud_world = depth_to_world_seg(depth_1,self.camera_intr,self.T_camera_world,self.plane_workspace)
        
        # compute the segmask for points above the box
        seg_point_cloud_cam = self.T_camera_world.inverse() * seg_point_cloud_world
        depth_2 = self.camera_intr.project_to_image(seg_point_cloud_cam)
        segmask = depth_2.to_binary()
        rgb_2 = rgb_1.mask_binary(segmask)
        # plot_depth(depth_2)
        print(seg_point_cloud_world.y_coords.min())
        seg2_point_cloud_world, _ = seg_point_cloud_world.box_mask(self.box_workspace)
        seg2_point_cloud_cam = self.T_camera_world.inverse() * seg2_point_cloud_world
        depth_3 = self.camera_intr.project_to_image(seg2_point_cloud_cam)
        segmask = depth_3.to_binary()
        rgb_3 = rgb_2.mask_binary(segmask)
        print (depth_3.data.shape)
        # depth_3._data[:,500:] = 0
        # plot_depth(depth_3)

        # ind = np.where(depth_3.data != 0)
        # center_i, center_j = np.mean(ind[0]), np.mean(ind[1])
        # height, width = np.max(ind[0])- np.min(ind[0]), np.max(ind[1])- np.min(ind[1])
        # # print(height, width, center_i, center_j)
        # depth_int = depth_3.crop(256, 256, center_i, center_j)
        # depth_4 = depth_int.resize((128,128), 'nearest')

        # return depth_4

        frame_string = "00"
        Plot_Image(rgb_1.data, self.base_path + "/rgb_1" + frame_string + ".png")
        Plot_Image(rgb_2.data, self.base_path + "/rgb_2" + frame_string + ".png")
        Plot_Image(rgb_3.data, self.base_path + "/rgb_3" + frame_string + ".png")
        Plot_Image(depth_1.data, self.base_path + "/depth_1" + frame_string + ".png")
        Plot_Image(depth_2.data, self.base_path + "/depth_2" + frame_string + ".png")
        Plot_Image(depth_3.data, self.base_path + "/depth_3" + frame_string + ".png")
        # Plot_Image(depth_4.data, self.base_path + "/depth_4" + frame_string + ".png")

        depth_3.save(self.base_path + "/depth_3" + frame_string + ".npz")
        seg2_point_cloud_world.save(self.base_path + "/pc_3" + frame_string + ".npz")
        # depth_4 = DepthImage.open(self.base_path + "/depth_3" + frame_string + ".npz")
        # print depth_3.data.max(), depth_4.data.max()
        # print depth_3.data[depth_3.data!=0].min(), depth_4.data[depth_4.data!=0].min()

    def process_clamshell_negative(self):
        img = self.sensor.read()
        rgb_1, depth_1 = img.color, img.depth

        seg_point_cloud_world = depth_to_world_seg(depth_1,self.camera_intr,self.T_camera_world,self.plane_workspace)
        
        # compute the segmask for points above the box
        seg_point_cloud_cam = self.T_camera_world.inverse() * seg_point_cloud_world
        depth_2 = self.camera_intr.project_to_image(seg_point_cloud_cam)
        segmask = depth_2.to_binary()
        rgb_2 = rgb_1.mask_binary(segmask)
        # plot_depth(depth_2)
        print(seg_point_cloud_world.y_coords.min())
        seg2_point_cloud_world, _ = seg_point_cloud_world.box_mask(self.negative_workspace)
        seg2_point_cloud_cam = self.T_camera_world.inverse() * seg2_point_cloud_world
        depth_3 = self.camera_intr.project_to_image(seg2_point_cloud_cam)
        segmask = depth_3.to_binary()
        rgb_3 = rgb_2.mask_binary(segmask)
        print (depth_3.data.shape)
        # depth_3._data[:,500:] = 0
        # plot_depth(depth_3)

        # ind = np.where(depth_3.data != 0)
        # center_i, center_j = np.mean(ind[0]), np.mean(ind[1])
        # height, width = np.max(ind[0])- np.min(ind[0]), np.max(ind[1])- np.min(ind[1])
        # # print(height, width, center_i, center_j)
        # depth_int = depth_3.crop(256, 256, center_i, center_j)
        # depth_4 = depth_int.resize((128,128), 'nearest')

        # return depth_4

        frame_string = "00"
        Plot_Image(rgb_1.data, self.base_path + "/rgb_1" + frame_string + ".png")
        Plot_Image(rgb_2.data, self.base_path + "/rgb_2" + frame_string + ".png")
        Plot_Image(rgb_3.data, self.base_path + "/rgb_3" + frame_string + ".png")
        Plot_Image(depth_1.data, self.base_path + "/depth_1" + frame_string + ".png")
        Plot_Image(depth_2.data, self.base_path + "/depth_2" + frame_string + ".png")
        Plot_Image(depth_3.data, self.base_path + "/depth_3" + frame_string + ".png")
        # Plot_Image(depth_4.data, self.base_path + "/depth_4" + frame_string + ".png")

        depth_3.save(self.base_path + "/depth_3" + frame_string + ".npz")
        seg2_point_cloud_world.save(self.base_path + "/pc_3" + frame_string + ".npz")
        # depth_4 = DepthImage.open(self.base_path + "/depth_3" + frame_string + ".npz")
        # print depth_3.data.max(), depth_4.data.max()
        # print depth_3.data[depth_3.data!=0].min(), depth_4.data[depth_4.data!=0].min()
        return seg2_point_cloud_world.mean().data

    def get_goal_img(self, project=False):
        points = np.load(self.base_path + "/hb/clamshell_positive/pc_300.npz")['arr_0']
        hb_pc = PointCloud(points, 'world')
        if project:
            z_translate = 0.5 - hb_pc.z_coords.max()
            hb_pc._data += np.array([[0],[0],[z_translate]])
            depth= world_to_image(hb_pc,self.camera_intr,self.T_camera_world)
        else:
            depth = DepthImage.open(self.base_path + "/hb/clamshell_positive/depth_300.npz")
        ind = np.where(depth.data != 0)
        center_i, center_j = (np.max(ind[0]) + np.min(ind[0])) / 2, (np.max(ind[1]) + np.min(ind[1])) / 2 
        height, width = np.max(ind[0])- np.min(ind[0]), np.max(ind[1])- np.min(ind[1])
        height, width = (height*11)/10, (width*11)/10
        height, width = max((height,width)), max((height,width))
        # print(height, width, center_i, center_j)
        depth_int = depth.crop(height, width, center_i, center_j)
        depth = depth_int.resize((128,128), 'nearest')
        # return demean_preprocess(depth.data)
        print(hb_pc.mean())
        return depth.data

    def check_urdf(self):
        img = self.sensor.read()
        rgb, depth = img.color, img.depth
        pc = self.camera_intr.deproject(depth)
        pc.remove_zero_points()
        pc, _ = pc.subsample(10)
        pc = self.T_camera_world * pc
        yaml_obj_loader = YamlObjLoader("/home/shivin/catkin_ws/src/ambidex/tests/cfg/")
        segment_robot(yaml_obj_loader('physical_yumi_no_jaw'), pc.data.T)


def inpaint_goal_img():
    points = DepthImage.open( "physical/cavities/" + "/hb/clamshell_positive/depth_300.npz")
    segmask = np.zeros(points.data.shape)
    for i,row in enumerate(points.data):
        indices = np.where(row != 0)[0]
        if len(indices) >= 1:
            min_idx, max_idx = np.min(indices), np.max(indices)
        else:
            min_idx, max_idx = 0,0
        for j in range(min_idx, max_idx+1):
            segmask[i][j] = 1
    x,y = np.where(segmask == 0)
    # points._data[x,y] = 5
    points = points.inpaint()
    points._data[x,y] = 0
    Plot_Image(points.data,  "physical/cavities/" + "/hb/inpaint.png")
        
if __name__ == "__main__":
    cavity = Cavity(plot=True)

    # cavity.capture_image_ideal()
    # cavity.capture_image()

    # cavity.process_clamshell_positive()
    cavity.process_clamshell_negative()
    # cavity.get_goal_img()

    # cavity.check_urdf()
    # inpaint_goal_img()
    # cavity.sensor.stop()