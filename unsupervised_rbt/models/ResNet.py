import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNSELU(nn.Sequential):
		def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True, dilation=1):
				padding = (kernel_size - 1) // 2
				super(ConvBNSELU, self).__init__(
						nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
						nn.BatchNorm2d(C_out),
						nn.SELU(inplace=True)
				)

class ConvBNReLU(nn.Sequential):
		def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True, dilation=1):
				padding = (kernel_size - 1) // 2
				super(ConvBNReLU, self).__init__(
						nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
						nn.BatchNorm2d(C_out),
						nn.ReLU(inplace=True)
				)

class ResnetBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, bias=False):
        super(ResnetBasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.conv1 = ConvBNReLU(C_in=1,C_out=64,kernel_size=3,stride=2, bias=False) #SE3 is 4->64, 7x7, stride 2
        # self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        # self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2 = ResnetBasicBlock(64,64,bias=False, stride=2)
    
    def forward(self, x):
        out = self.conv1(x)
        # out = self.pool1(out)
        out = self.conv2(out)
        return out

class ResNetSiameseNetwork(nn.Module):
    def __init__(self, split_resnet = False):
        super(ResNetSiameseNetwork, self).__init__()
        self.resnet = FeatureNet()
        if split_resnet:
            self.resnet2 = FeatureNet()
        self.split_resnet = split_resnet

        self.conv1 = ConvBNReLU(128,256,stride=2)
        self.conv2 = ResnetBasicBlock(256,256)
        self.conv3 = ConvBNReLU(256,256,stride=2)
        self.conv4 = ResnetBasicBlock(256,256)
        self.conv5 = ConvBNReLU(256,256,stride=2)
        self.conv6 = ResnetBasicBlock(256,256)
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.final_fc = nn.Linear(4096,4)

    def forward(self, input1, input2):
        output1 = self.resnet(input1)
        output2 = self.resnet2(input2) if self.split_resnet else self.resnet(input2)
        output_concat = torch.cat((output1, output2), 1)

        output = self.conv1(output_concat)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.pool(output)
        output = output.reshape(output.shape[0],-1)
                
        output = self.final_fc(output)
        # print(output)
        output = F.normalize(output)
        return output

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, strides=[1,1]):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=strides[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if strides[0] + strides[1] != 2 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=strides[0]*strides[1], bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_output=200, image_dim = 128):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.n_blocks = num_blocks
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if image_dim == 128:
            if self.n_blocks == 1:
                self.layer1 = self._make_layer(block, 64, strides=[2,1])
                self.linear = nn.Linear(4096, num_output)
            elif self.n_blocks == 2:
                self.layer1 = self._make_layer(block, 64, strides=[1,1])
                self.layer2 = self._make_layer(block, 64, strides=[2,1])
                self.linear = nn.Linear(1024, num_output)
            elif self.n_blocks == 3:
                self.layer1 = self._make_layer(block, 64, strides=[1,1])
                self.layer2 = self._make_layer(block, 64, strides=[2,1])
                self.layer3 = self._make_layer(block, 64, strides=[2,1])
                self.linear = nn.Linear(1024, num_output)
            else:
                print("Error: number of blocks not in ResNet specification")
                assert False
        elif image_dim == 256:
            if self.n_blocks == 1:
                self.layer1 = self._make_layer(block, 64, strides=[2,2])
                self.linear = nn.Linear(4096, num_output)
            elif self.n_blocks == 2:
                self.layer1 = self._make_layer(block, 64, strides=[2,1])
                self.layer2 = self._make_layer(block, 64, strides=[2,1])
                self.linear = nn.Linear(4096, num_output)
            elif self.n_blocks == 3:
                self.layer1 = self._make_layer(block, 64, strides=[1,1])
                self.layer2 = self._make_layer(block, 64, strides=[2,1])
                self.layer3 = self._make_layer(block, 64, strides=[2,1])
                self.linear = nn.Linear(4096, num_output)
            else:
                print("Error: number of blocks not in ResNet specification")
                assert False

    def _make_layer(self, block, planes, strides):
        return nn.Sequential(block(self.in_planes, planes, strides))

    def forward_no_linear(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.maxpool(out)
        out = self.layer1(out)
        if self.n_blocks == 1:
            return out
        elif self.n_blocks == 2:  
            out = self.layer2(out)
            return out
        elif self.n_blocks == 3:
            out = self.layer2(out)
            out = self.layer3(out)
            return out
        return out

    def forward(self, x):
        out = self.forward_no_linear(x)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

class LinearEmbeddingClassifier(nn.Module):
    def __init__(self, config, num_classes, embed_dim=200, dropout=False, init=False):
        super(LinearEmbeddingClassifier, self).__init__()
        siamese = ResNetSiameseNetwork(config['pred_dim'], n_blocks=1, embed_dim=embed_dim, dropout=dropout, norm=False)
        if init:
            print('------------- Loaded self-supervised model -------------')
            # siamese.load_state_dict(torch.load(config['unsup_model_save_dir']))
            new_state_dict = siamese.state_dict()
            load_params = torch.load(config['unsup_model_path'])
            load_params_new = load_params.copy()
            for layer_name in load_params:
                if not layer_name.startswith(('resnet')):
                    # print("deleting layer", layer_name)
                    del load_params_new[layer_name]
            new_state_dict.update(load_params_new)
            siamese.load_state_dict(new_state_dict)

        self.resnet = siamese.resnet
        self.fc_1 = nn.Linear(embed_dim, 1000) # was 200 before (but 50 achieves same result for rotation prediction)
        self.fc_2 = nn.Linear(1000, 1000)
        self.final_fc = nn.Linear(1000, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input1):
        output = self.resnet(input1)
        output = self.dropout(output)
        output = self.dropout(F.relu(self.fc_1(output)))
        output = self.dropout(F.relu(self.fc_2(output)))
        return self.final_fc(output)
    
class ResNetObjIdPred(nn.Module):
    def __init__(self, transform_pred_dim, dropout=False, embed_dim=200, n_blocks = 4):
        super(ResNetObjIdPred, self).__init__()
        blocks = [item for item in [1] for i in range(n_blocks)]
        self.resnet = ResNet(BasicBlock, blocks, embed_dim, dropout=False)   # [1,1,1,1]
        self.fc_1 = nn.Linear(embed_dim, 1000) # was 200 before (but 50 achieves same result)
        self.fc_2 = nn.Linear(1000, 1000)
        self.final_fc = nn.Linear(1000, transform_pred_dim)  
        self.dropout = nn.Dropout(0.2)

    def forward(self, input1):
        output = self.resnet(input1)
        output = self.dropout(F.relu(self.fc_1(output)))
        output = self.dropout(F.relu(self.fc_2(output)))
        return self.final_fc(output)
