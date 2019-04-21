import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

#from autolab_core import YamlConfig, RigidTransform
from unsupervised_rbt import TensorDataset
import itertools
#from perception import DepthImage, RgbdImage

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

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = ResNet(BasicBlock, [1,1,1,1], 100)
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
        if stride != 1 or in_planes != self.expansion*planes: # if different feature size after
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
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.linear = nn.Linear(6912, num_output)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # what is this supposed to do?
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
    #N_train = int(train_frac*dataset.num_datapoints)
    #n_train_steps = N_train//batch_size
    train_loss = 0
    correct = 0
    total = 0
    
    # get train indices
    train_indices = dataset.split('train')[0]
    N_train = len(train_indices)
    n_train_steps = N_train//batch_size
    
    # run over all batches (n_train_steps)
    for step in tqdm(range(n_train_steps)):
        
        # Step 1: get and preprocess depth images for batches
        batch = dataset.get_item_list(train_indices[step*batch_size : (step+1)*batch_size])
        # quantization from 0-1 float to 255 ints
        depth_image1 = (batch["depth_image1"] * 255).astype(int) # why *255? why as int?
        depth_image2 = (batch["depth_image2"] * 255).astype(int)
        
        im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
        im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
        transform_batch = Variable(torch.from_numpy(batch["transform"].astype(int))).to(device)
        
        # Step 2: feed through the model
        optimizer.zero_grad()
        pred_transform = model(im1_batch, im2_batch) # no softmax?
        _, predicted = torch.max(pred_transform, 1) # indice of the max
        correct += (predicted == transform_batch).sum().item() # total number of correct
        total += transform_batch.size(0) # total numbers (because might have uneven batch sizes)
        
        loss = criterion(pred_transform, transform_batch)
        loss.backward()
        train_loss += loss.item() # accumulate the train loss
        optimizer.step()
    
    # total classification accuracy once over the whole training set
    class_acc = 100 * correct/total
    # return average loss and average classification accuracy
    return train_loss/n_train_steps, class_acc


def test(dataset):
    model.eval()
    
    # initialize all tracking variables
    test_loss = 0
    correct = 0
    total = 0

    test_indices = dataset.split('train')[1] # train [1] is the test set?
    N_test = len(test_indices)
    n_test_steps = N_test // batch_size
    with torch.no_grad():
        for step in tqdm(range(n_test_steps)):
            batch = dataset.get_item_list(test_indices[step*batch_size : (step+1)*batch_size])
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
        img = img / 2 + 0.5     # unnormalize ??
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0))) # why?
        plt.show()
    with torch.no_grad():
        imshow(torchvision.utils.make_grid(model.resnet.conv1.weight)) # why that layer?
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_train = not args.test
    losses_f_name = "results/losses_free_space.p"
    loss_plot_f_name = "plots/losses_free_space.png"
    transform_pred_dim = 4

    if run_train:
        train_frac = 0.8
        batch_size = 8
        dataset = TensorDataset.open("/nfs/diskstation/projects/unsupervised_rbt/z_axis_angle_pred")
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

        
