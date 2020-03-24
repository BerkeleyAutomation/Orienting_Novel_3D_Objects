import torch
import kornia

class ShapeMatchLoss(torch.nn.Module):
    """
    Shape-Match Loss function.
    Based on PoseCNN
    """

    def __init__(self):
        super(ShapeMatchLoss, self).__init__()

    @staticmethod
    def PointCloudDistance(x,y): # for example, x = (batch,2025,3), y = (batch,2048,3)
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

        chamfer_distance = torch.mean(x_y_row, 1)  # batch
        chamfer_distance = torch.mean(chamfer_distance) # loss
        return chamfer_distance

    def forward(self, predquat, gtquat, points):
        predrot = kornia.quaternion_to_rotation_matrix(predquat)
        gtrot = kornia.quaternion_to_rotation_matrix(gtquat)
        # print(predrot)
        # print(gtrot)
        predpts = torch.bmm(predrot, points)
        gtpts = torch.matmul(gtrot, points)
        # print(predpts)
        # print(gtpts)
        return self.PointCloudDistance(gtpts, predpts)

if __name__ == "__main__":
    points = [[0,0,1],[0,1,0],[1,0,0],[1,1,0]]
    points = torch.Tensor(points).t()
    points = torch.unsqueeze(points,0)
    print(points.size())
    predquat = torch.Tensor([[0,0,0.5 ** 0.5,0.5 ** 0.5]]) #90 degree z-axis
    gtquat = torch.Tensor([[0,0,1,0]]) # 180 degree z-axis
    lossfn = ShapeMatchLoss()
    loss = lossfn.forward(predquat,gtquat,points)
    print(loss.item())