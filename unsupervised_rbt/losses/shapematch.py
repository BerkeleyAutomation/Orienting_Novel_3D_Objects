import torch
from unsupervised_rbt.losses import kornia
import pickle 
import numpy as np

class ShapeMatchLoss(torch.nn.Module):
    """
    Shape-Match Loss function.
    Based on PoseCNN
    """

    def __init__(self):
        super(ShapeMatchLoss, self).__init__()

    @staticmethod
    def PointCloudDistance(x,y): # for example, x = (batch,2025,3), y = (batch,2048,3) ONLY works if # points in point cloud is the same across batches
        # code from https://github.com/YanWei123/Pytorch-implementation-of-FoldingNet-encoder-and-decoder-with-graph-pooling-covariance-add-quanti
        """x is ground truth point cloud, y is rotated point cloud"""
        x_size = x.size() 
        y_size = y.size()
        assert (x_size[0] == y_size[0])
        assert (x_size[2] == y_size[2])
        x = torch.unsqueeze(x, 1)  # x = batch,1,2025,3
        y = torch.unsqueeze(y, 2)  # y = batch,2048,1,3

        x = x.repeat(1, y_size[1], 1, 1)  # x = batch,2048,2025,3
        y = y.repeat(1, 1, x_size[1], 1)  # y = batch,2048,2025,3

        x_y = x - y
        x_y = torch.pow(x_y, 2)  # x_y = batch,2048,2025,3
        x_y = torch.sum(x_y, 3, keepdim=True)  # x_y = batch,2048,2025,1
        x_y = torch.squeeze(x_y, 3)  # x_y = batch,2048,2025
        x_y_row, _ = torch.min(x_y, 1)  # x_y_row = batch,2025
        # print(x_y_row)
        chamfer_distance = torch.mean(x_y_row, 1)  # batch
        chamfer_distance = torch.mean(chamfer_distance) # loss
        return chamfer_distance

    def forward(self, predquat, gtquat, points): # points should be of shape (batch x 3 x npoints)
        predrot = kornia.quaternion_to_rotation_matrix(predquat)
        gtrot = kornia.quaternion_to_rotation_matrix(gtquat)
        # print(predrot)
        # print(gtrot)
        predpts = torch.bmm(predrot, points)
        gtpts = torch.matmul(gtrot, points)
        # print(predpts)
        # print(gtpts)
        return self.PointCloudDistance(torch.transpose(gtpts,1,2), torch.transpose(predpts,1,2))

if __name__ == "__main__":
    points = [[0,0,1],[0,1,0],[1,0,0],[1,1,0]]
    points = torch.Tensor(points).t()
    points = torch.unsqueeze(points,0)
    print(points.size())
    predquat = torch.Tensor([[0,0,0.5 ** 0.5,0.5 ** 0.5]]) #90 degree z-axis
    gtquat = torch.Tensor([[0,0,1,0]]) # 180 degree z-axis
    lossfn = ShapeMatchLoss()
    loss = lossfn.forward(predquat,gtquat,points)
    print("loss should be:", (0 + 2 + 0 + 1)/4)
    print(loss.item())

    point_clouds = pickle.load(open("cfg/tools/point_clouds", "rb"))
    scales = pickle.load(open("cfg/tools/scales", "rb"))
    # num_pts = []
    # for k,v in point_clouds.items():
    #     cur = v.shape[1]
    #     num_pts.append(cur)
    #     if k == 372:
    #         print(v.shape) 
    # num_pts = np.array(num_pts)
    # print("Number of points above 100 is:", np.sum(num_pts >= 100))
    # print("Number of points above 150 is:", np.sum(num_pts >= 150))
    # print("Number of points above 200 is:", np.sum(num_pts >= 200))
    # print("Number of points above 250 is:", np.sum(num_pts >= 250))
    # print("Number of points above 300 is:", np.sum(num_pts >= 300))
    print(np.mean(list(scales.values())))
    obj_id = 11
    rot_mtx = np.array([[ 0.0869703 , -0.996178  , -0.0080988 ],
                        [-0.23559828, -0.0284663 ,  0.9714335 ],
                        [-0.96795124, -0.0825778 , -0.23717354]])
    points = torch.Tensor([rot_mtx @ point_clouds[obj_id]])
    print(points.size())
    predquat =  torch.Tensor([[-0.49963564, -0.5105355,  -0.49345765, -0.4962029 ]]) #this quat turns shoe 90 degrees
    predquat = torch.Tensor([[ -0.00886435,  -0.00668257, 0.00629114,  -0.9999186 ]]) # negative of the quaternion
    gtquat = torch.Tensor([[ 0.00886435,  0.00668257, -0.00629114,  0.9999186 ]]) #almost no rotation, shoe facing forward
    loss = lossfn.forward(predquat,gtquat,points)
    print(loss.item())
