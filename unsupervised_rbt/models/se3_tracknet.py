#
# Authors: Bowen Wen
# Contact: wenbowenxjtu@gmail.com
# Created in 2020
#
# Copyright (c) Rutgers University, 2020 All rights reserved.
#
# Wen, B., C. Mitash, B. Ren, and K. E. Bekris. "se (3)-TrackNet:
# Data-driven 6D Pose Tracking by Calibrating Image Residuals in
# Synthetic Domains." In IEEE/RSJ International Conference on Intelligent
# Robots and Systems (IROS). 2020.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the PRACSYS, Bowen Wen, Rutgers University,
#       nor the names of its contributors may be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import os,sys
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from torch import optim
import cv2

class Se3TrackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rot_dim = 4
        self.resnet = FeatureNet()

        self.convAB1 = ConvBNReLU(128,256,kernel_size=3,stride=2)
        self.convAB2 = ResnetBasicBlock(256,256,bias=False)
        self.rot_conv1 = ConvBNReLU(256,256,kernel_size=3,stride=2)
        self.rot_conv2 = ResnetBasicBlock(256,256,bias=False)
        self.rot_pool1 = nn.AdaptiveAvgPool2d((4,4))
        self.rot_out = nn.Linear(4096,self.rot_dim)

    def forward(self, A, B):
        batch_size = A.shape[0]
        output = {}

        a = self.resnet(A)
        b = self.resnet(B)

        ab = torch.cat((a,b),1)
        ab = self.convAB1(ab)
        ab = self.convAB2(ab)
        output['feature'] = ab

        rot = self.rot_conv1(ab)
        rot = self.rot_conv2(rot)
        # print(rot.shape)
        rot = self.rot_pool1(rot)
        rot = rot.reshape(batch_size,-1)
        rot = self.rot_out(rot)
        rot = F.normalize(rot)
        output['rot'] = rot

        return rot

    def loss(self, predictions, targets):
        output = {}
        trans_loss = nn.MSELoss()(predictions[0].float(), targets[0].float())
        rot_loss = nn.MSELoss()(predictions[1].float(), targets[1].float())
        output['trans'] = trans_loss
        output['rot'] = rot_loss

        return output

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.conv1 = ConvBNReLU(C_in=1,C_out=64,kernel_size=3,stride=2, bias=False) #SE3 is 4->64, 7x7, stride 2
        # self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2 = ResnetBasicBlock(64,64,bias=False, stride=2)
    
    def forward(self, x):
        out = self.conv1(x)
        # out = self.pool1(out)
        out = self.conv2(out)
        return out

class ConvBN(nn.Sequential):
        def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,dilation=1,):
                padding = (kernel_size - 1) // 2
                super(ConvBN, self).__init__(
                        nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
                        nn.BatchNorm2d(C_out),
                )

class ConvBNReLU(nn.Sequential):
        def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,):
                padding = (kernel_size - 1) // 2
                super(ConvBNReLU, self).__init__(
                        nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias),
                        nn.BatchNorm2d(C_out),
                        nn.ReLU(inplace=False)
                )

class ConvBNSELU(nn.Sequential):
        def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=False,dilation=1,):
                padding = (kernel_size - 1) // 2
                super(ConvBNSELU, self).__init__(
                        nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
                        nn.BatchNorm2d(C_out),
                        nn.SELU(inplace=True)
                )

class ConvPadding(nn.Module):
    def __init__(self,C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,dilation=1):
        super(ConvPadding, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation)

    def forward(self,x):
        return self.conv(x)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)

class ResnetBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, bias=False):
        super().__init__()
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
