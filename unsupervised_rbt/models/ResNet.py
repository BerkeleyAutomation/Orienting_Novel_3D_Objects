import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetSiameseNetwork(nn.Module):
    def __init__(self, transform_pred_dim, dropout=False):
        super(ResNetSiameseNetwork, self).__init__()
        self.resnet = ResNet(BasicBlock, [1,1,1,1], 200, dropout=False)
        self.fc_1 = nn.Linear(200*2, 4000)
        self.fc_2 = nn.Linear(4000, 4000)
        self.final_fc = nn.Linear(4000, transform_pred_dim)  
        self.dropout = nn.Dropout(0.2)

    def forward(self, input1, input2):
        output1 = self.resnet(input1)
        output2 = self.resnet(input2)
        output_concat = torch.cat((output1, output2), 1)
        output = self.dropout(F.relu(self.fc_1(output_concat)))
        output = self.dropout(F.relu(self.fc_2(output)))
        return self.final_fc(output)
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(self.bn1(x)))
        out = self.conv2(self.bn2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_output=10, dropout=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # dropout
        if dropout:
            self.p = 0.1
        else:
            self.p = 0
        self.dropout2d = torch.nn.Dropout2d(p=self.p)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.linear = nn.Linear(1024, num_output)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_no_linear(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.dropout2d(out)
        out = self.layer2(out)
        out = self.dropout2d(out)
        out = self.layer3(out)
        out = self.dropout2d(out)
        return self.layer4(out)

    def forward(self, x):
        out = self.forward_no_linear(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
