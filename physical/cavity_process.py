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

class Cavity(BaseTask):
    def __init__(self, sensor=None, plot_images=True):
        self.init_camera(sensor)
        self.init_workspace()
        self.init_attributes(plot_images)

    def init_workspace(self):
        phoxi_config = YamlConfig("physical/cfg/tools/colorized_phoxi.yaml")
        self.plane_workspace = Box(np.array(phoxi_config['plane_workspace']['min_pt']),
                            np.array(phoxi_config['plane_workspace']['max_pt']),
                            frame='world')

        self.positive_workspace = Box(np.array(phoxi_config['positive_workspace']['min_pt']),
                            np.array(phoxi_config['positive_workspace']['max_pt']),
                            frame='world')

        self.negative_workspace = Box(np.array(phoxi_config['negative_workspace']['min_pt']),
                            np.array(phoxi_config['negative_workspace']['max_pt']),
                            frame='world')

        self.elevated_negative_workspace = Box(np.array(phoxi_config['elevated_negative_workspace']['min_pt']),
                            np.array(phoxi_config['elevated_negative_workspace']['max_pt']),
                            frame='world')

    def init_attributes(self, plot_images):
        self.base_path = "physical/cavities/"
        self.plot_images = plot_images

    def capture_image_ideal(self, center_i=None, center_j=None, frame=0):
        # logger.info('Capturing depth image' + str(frame))
        img = self.sensor.read()
        rgb_1, depth_1 = img.color, img.depth

        depth_2, ideal_T_world_phoxi = self.top_down_img(depth_1)
        seg_point_cloud_world = depth_to_world_seg(depth_2,self.camera_intr,ideal_T_world_phoxi,self.plane_workspace)
        
        depth_3 = world_to_image(seg_point_cloud_world,self.camera_intr,ideal_T_world_phoxi)
        segmask = depth_3.to_binary()

        z_thresh = np.mean(seg_point_cloud_world.data,1)[2]
        seg2_point_cloud_world = seg_point_cloud_world[seg_point_cloud_world.data[2] < z_thresh]
        depth_4 = world_to_image(seg2_point_cloud_world,self.camera_intr,ideal_T_world_phoxi)
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

        # depth_4 = self.crop_image(depth_3)
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
        seg2_point_cloud_world, _ = seg_point_cloud_world.box_mask(self.positive_workspace)
        seg2_point_cloud_cam = self.T_camera_world.inverse() * seg2_point_cloud_world
        depth_3 = self.camera_intr.project_to_image(seg2_point_cloud_cam)
        segmask = depth_3.to_binary()
        rgb_3 = rgb_2.mask_binary(segmask)
        print ("Positive Min Bounds:", seg2_point_cloud_world.data.min(1), "Max Bounds:", seg2_point_cloud_world.data.max(1))
        print ("Centroid:", seg2_point_cloud_world.mean().data)
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

    def get_goal_img(self, obj_id, project=False):
        folder_paths = {5: "/hb/positive_0/", 6:"/ohb/positive_0/",7:"/sink_handle/positive_0/"}
        folder_path = self.base_path + folder_paths[obj_id]
        points = np.load(folder_path + "pc_300.npz")['arr_0']
        hb_pc = PointCloud(points, 'world')
        if project:
            z_translate = 0.5 - hb_pc.z_coords.max()
            hb_pc._data += np.array([[0],[0],[z_translate]])
            depth= world_to_image(hb_pc,self.camera_intr,self.T_camera_world)
        else:
            depth = DepthImage.open(folder_path + "depth_300.npz")

        depth = crop_image(depth, bb_center=True, bb_slack=11)
        # return demean_preprocess(depth.data)
        print(hb_pc.mean())
        return depth.data, hb_pc

    def process_clamshell_negative(self, elevated = False):
        img = self.sensor.read()
        rgb_1, depth_1 = img.color, img.depth

        seg_point_cloud_world = depth_to_world_seg(depth_1,self.camera_intr,self.T_camera_world,self.plane_workspace)
        
        neg_workspace = self.elevated_negative_workspace if elevated else self.negative_workspace

        # compute the segmask for points above the box
        depth_2 = world_to_image(seg_point_cloud_world,self.camera_intr,self.T_camera_world)
        segmask = depth_2.to_binary()
        rgb_2 = rgb_1.mask_binary(segmask)
        # plot_depth(depth_2)
        print(seg_point_cloud_world.y_coords.min())
        seg2_point_cloud_world, _ = seg_point_cloud_world.box_mask(neg_workspace)
        depth_3 = world_to_image(seg2_point_cloud_world, self.camera_intr,self.T_camera_world)
        segmask = depth_3.to_binary()
        rgb_3 = rgb_2.mask_binary(segmask)
        print (depth_3.data.shape)
        # depth_3._data[:,500:] = 0
        # plot_depth(depth_3)

        # depth_4 = crop_image(depth_3)
        # return depth_4

        depth_4 = remove_junk(depth_3, thresh = 1000.0)

        seg3_point_cloud_world = depth_to_world_seg(depth_4, self.camera_intr, self.T_camera_world,neg_workspace)
        frame_string = "00"
        if self.plot_images:
            Plot_Image(rgb_1.data, self.base_path + "/rgb_1" + frame_string + ".png")
            Plot_Image(rgb_2.data, self.base_path + "/rgb_2" + frame_string + ".png")
            Plot_Image(rgb_3.data, self.base_path + "/rgb_3" + frame_string + ".png")
            Plot_Image(depth_1.data, self.base_path + "/depth_1" + frame_string + ".png")
            Plot_Image(depth_2.data, self.base_path + "/depth_2" + frame_string + ".png")
            Plot_Image(depth_3.data, self.base_path + "/depth_3" + frame_string + ".png")
            Plot_Image(depth_4.data, self.base_path + "/depth_4" + frame_string + ".png")

        depth_4.save(self.base_path + "/depth_3" + frame_string + ".npz")
        seg3_point_cloud_world.save(self.base_path + "/pc_4" + frame_string + ".npz")
        # depth_4 = DepthImage.open(self.base_path + "/depth_3" + frame_string + ".npz")
        # print depth_3.data.max(), depth_4.data.max()
        # print depth_3.data[depth_3.data!=0].min(), depth_4.data[depth_4.data!=0].min()
        print ("Positive Min Bounds:", seg3_point_cloud_world.data.min(1), "Max Bounds:", seg3_point_cloud_world.data.max(1))
        print ("Centroid:", seg3_point_cloud_world.mean().data)
        return seg3_point_cloud_world

    def rotate_negative(self, pc):
        com = pc.mean().data
        rot = RigidTransform.rotation_from_axis_and_origin([1,0,0], com, np.pi, to_frame='world')
        rot_pc = rot * pc
        depth = world_to_image(rot_pc, self.camera_intr,self.T_camera_world)
        depth_2 = remove_junk(depth, 1000.0)
        depth_2 = crop_image(depth_2,bb_center=True,bb_slack=11)

        if self.plot_images:
            Plot_Image(depth.data, self.base_path + "/rot_negative.png")
            Plot_Image(depth_2.data, self.base_path + "/rot_negative_crop.png")
        return depth_2.data, com

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
    cavity = Cavity(plot_images=True)

    # cavity.capture_image_ideal()
    # cavity.capture_image()

    # cavity.process_clamshell_positive()
    # cavity.get_goal_img()

    # cavity.process_clamshell_negative()

    pc = cavity.process_clamshell_negative(elevated=True)
    cavity.rotate_negative(pc)

    # cavity.check_urdf()
    # inpaint_goal_img()
    # cavity.sensor.read()