import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: modelled after the paper while leaving out some pool layers and adapting kernel sizes for our problem https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf

class ContextSiameseNetwork(nn.Module):
    def __init__(self, transform_pred_dim):
        super(ContextSiameseNetwork, self).__init__()
        self.base_net = Base_Network()
        self.fc7 = nn.Linear(2*4096, 4096)
        self.fc8 = nn.Linear(4096, 4096)
        self.final_fc = nn.Linear(4096, transform_pred_dim)

    def forward(self, input1, input2):
        output1 = self.base_net(input1)
        output2 = self.base_net(input2)
        output_concat = torch.cat((output1, output2), 1)
        out = self.fc7(output_concat)
        out = self.fc8(out)
        return self.final_fc(out)
    
    
class Base_Network(nn.Module):
    def __init__(self, in_planes=1):
        super(Base_Network, self).__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_planes, 96, kernel_size=6, stride=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 384, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(384)
        self.conv3 = nn.Conv2d(384, 384, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1)
        self.fc6 = nn.Linear(9216, 4096)
        
    def forward(self, x):
        # N 1, 128, 128
        out = F.relu(self.conv1(self.bn0(x)))
        # N 96, 62, 62
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        # N 96, 30, 30
        out = F.relu(self.conv2(self.bn1(out)))
        # N 384, 14, 14
        out = self.conv3(out)
        # N 384, 10, 10
        out = self.conv4(out)
        # N 384, 8, 8
        out = self.conv5(out)
        # N 384, 6, 6
        out = out.view(out.size(0), -1)
        # N 9216
        out = self.fc6(out)
        # N 4096
        return out