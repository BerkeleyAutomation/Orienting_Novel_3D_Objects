import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

from autolab_core import YamlConfig, RigidTransform
from unsupervised_rbt import TensorDataset
import itertools
from perception import DepthImage, RgbdImage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: Make this a real architecture, this is just a minimum working example for now
# TODO: Convert transform to euler angles, right now just flattening transform
# and learning to predict that
# TODO: Improve batching speed/data loading, its still kind of slow rn
# TODO: Clean up so parameters not defined in __main__ but instead defined in config yaml

n_filters = 4
# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(1, n_filters, kernel_size=7, stride=2, padding=3),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(n_filters),
#             
#             nn.Conv2d(n_filters, n_filters, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(n_filters),
#             nn.MaxPool2d(2,2),
# 
#             nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(n_filters),
#             nn.MaxPool2d(2,2)
#             
# #             nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=2, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.BatchNorm2d(n_filters),
# #             nn.MaxPool2d(2,2)
#         )
#         # self.fc1 = nn.Sequential(
#         #     nn.Linear(n_filters*108, 100),
#         #     nn.ReLU(inplace=True),
#         #     nn.Linear(100, 100),
#         #     nn.ReLU(inplace=True)
#         # )
#         # 
#         # self.final_fc = nn.Linear(200, transform_pred_dim)
#         self.final_fc = nn.Linear(n_filters*108*2, transform_pred_dim)
# 
#     def forward_once(self, x):
#         output = self.cnn1(x)
#         output = output.view(output.size()[0], -1)
#         #output = self.fc1(output)
#         return output
# 
#     def forward(self, input1, input2):
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)
#         output_concat = torch.cat((output1, output2), 1)
#         output_final = self.final_fc(output_concat)
#         return output_final

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = ResNet(BasicBlock, [2,2,2,2], 100)
        self.final_fc = nn.Linear(100*2, transform_pred_dim)

    def forward(self, input1, input2):
        output1 = self.resnet(input1)
        output2 = self.resnet(input2)
        output_concat = torch.cat((output1, output2), 1)
        return self.final_fc(output_concat)

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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_output=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.linear = nn.Linear(512*block.expansion, num_output)

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
        out = self.layer2(out)
        out = self.layer3(out)
        return self.layer4(out)

    def forward(self, x):
        out = self.forward_no_linear(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def train(dataset):
    model.train()
    N_train = int(train_frac*dataset.num_datapoints)
    n_train_steps = N_train//batch_size
    train_loss = 0
    correct = 0
    total = 0
    
    #train_indices = dataset.split('train')[0]
    #N_train = len(train_indices)
    #n_train_steps = N_train//batch_size
    for step in tqdm(range(n_train_steps)):
        batch = dataset[step*batch_size : step*batch_size+batch_size]
        #batch = dataset.get_item_list(train_indices[step*batch_size : (step+1)*batch_size])
        depth_image1 = (batch["depth_image1"] * 255).astype(int)
        depth_image2 = (batch["depth_image2"] * 255).astype(int)
        im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
        im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
        transform_batch = Variable(torch.from_numpy(batch["transform"].astype(int))).to(device)
        optimizer.zero_grad()
        pred_transform = model(im1_batch, im2_batch)
        _, predicted = torch.max(pred_transform, 1)
        correct += (predicted == transform_batch).sum().item()
        total += transform_batch.size(0)
        
        loss = criterion(pred_transform, transform_batch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    class_acc = 100 * correct/total
    return train_loss/n_train_steps, class_acc

def test(dataset):
    model.eval()
    N_train = int(train_frac*dataset.num_datapoints)
    N_test = int( (1 - train_frac)*dataset.num_datapoints)
    n_test_steps = N_test//batch_size
    test_loss = 0
    correct = 0
    total = 0

    #test_indices = dataset.split('train')[1]
    #N_test = len(test_indices)
    #n_test_steps = N_test // batch_size
    with torch.no_grad():
        for step in tqdm(range(n_test_steps)):
            batch = dataset[step*batch_size + N_train : step*batch_size+batch_size + N_train]
            #batch = dataset.get_item_list(test_indices[step*batch_size : (step+1)*batch_size])
            depth_image1 = (batch["depth_image1"] * 255).astype(int)
            depth_image2 = (batch["depth_image2"] * 255).astype(int)
            im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
            im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
            transform_batch = Variable(torch.from_numpy(batch["transform"].astype(int))).to(device)
            pred_transform = model(im1_batch, im2_batch)
            _, predicted = torch.max(pred_transform, 1)
            correct += (predicted == transform_batch).sum().item()
            total += transform_batch.size(0)
            
            loss = criterion(pred_transform, transform_batch)
            test_loss += loss.item()
       
    class_acc = 100 * correct/total
    return test_loss/n_test_steps, class_acc

def display_conv_layers(model):
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    with torch.no_grad():
        imshow(torchvision.utils.make_grid(model.cnn1[0].weight))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_train = not args.test
    losses_f_name = "results/losses_free_space.p"
    loss_plot_f_name = "plots/losses_free_space.png"
    transform_pred_dim = 6

    if run_train:
        train_frac = 0.8
        batch_size = 4
        dataset = TensorDataset.open("/nfs/diskstation/projects/rbt_2/")
        im_shape = dataset[0]["depth_image1"].shape[:-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SiameseNetwork()
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        num_epochs = 100 
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        for epoch in range(num_epochs):
            train_loss, train_acc = train(dataset)
            test_loss, test_acc = test(dataset)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print("Epoch %d, Train Loss = %f, Train Acc = %.2f %%, Test Loss = %f, Test Acc = %.2f %%" % (epoch, train_loss, train_acc, test_loss, test_acc))
            pickle.dump({"train_loss" : train_losses, "train_acc" : train_accs, "test_loss" : test_losses, "test_acc" : test_accs}, open( losses_f_name, "wb"))
            torch.save(model.state_dict(), "models/rb_net_free_space.pt")
            
    else:
        model = SiameseNetwork()
        model.load_state_dict(torch.load("models/rb_net_free_space.pt"))
        display_conv_layers(model)

        losses = pickle.load( open( losses_f_name, "rb" ) )
        train_returns = np.array(losses["train_loss"])
        test_returns = np.array(losses["test_loss"])
        train_accs = np.array(losses["train_acc"])
        test_accs = np.array(losses["test_acc"])
        
        plt.plot(np.arange(len(train_returns)) + 1, train_returns, label="Training Loss")
        plt.plot(np.arange(len(test_returns)) + 1, test_returns, label="Testing Loss")
        plt.xlabel("Training Iteration")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.legend(loc='best')
        plt.savefig(loss_plot_f_name)
        plt.close()
        
        plt.plot(np.arange(len(train_accs)) + 1, train_accs, label="Training Acc")
        plt.plot(np.arange(len(test_accs)) + 1, test_accs, label="Testing Acc")
        plt.xlabel("Training Iteration")
        plt.ylabel("Classification Accuracy")
        plt.title("Training Curve")
        plt.legend(loc='best')
        plt.savefig(loss_plot_f_name)
        plt.close()

        
