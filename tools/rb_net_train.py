import numpy as np
import argparse
import os

from autolab_core import YamlConfig, RigidTransform
from unsupervised_rbt import TensorDataset
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer2D as vis2d
import itertools
from perception import DepthImage, RgbdImage

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt

# TODO: Make this a real architecture, this is just a minimum working example for now
# TODO: Convert transform to euler angles, right now just flattening transform
# and learning to predict that
# TODO: Improve batching speed/data loading, its still kind of slow rn
# TODO: Clean up so parameters not defined in __main__ but instead defined in config yaml

n_filters = 64
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_filters),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_filters),
            nn.MaxPool2d(2,2),

            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_filters),
            nn.MaxPool2d(2,2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(n_filters*5*6, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True)
        )
        
        self.final_fc = nn.Linear(1000, transform_pred_dim)

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output_concat = torch.cat((output1, output2), 1)
        output_final = self.final_fc(output_concat)
        return output_final

def train(epoch, dataset):
    model.train()
    train_loss = 0
    N_train = int(train_frac*dataset.num_datapoints)
    n_train_steps = N_train//batch_size
    
    print(n_train_steps)
    
    for step in range(n_train_steps):
        print("Step: " + str(step))
        batch = dataset[step*batch_size : step*batch_size+batch_size]
        im1_batch = Variable(torch.from_numpy(batch["depth_image1"]).float()).to(device)
        im2_batch = Variable(torch.from_numpy(batch["depth_image2"]).float()).to(device)
        transform_batch = Variable(torch.from_numpy(batch["transform"]).float()).to(device)
        optimizer.zero_grad()
        pred_transform = model(im1_batch, im2_batch)
        loss = criterion(pred_transform, transform_batch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    return train_loss/n_train_steps

def test(epoch, dataset):
    model.eval()
    test_loss = 0
    N_train = int(train_frac*dataset.num_datapoints)
    N_test = int( (1 - train_frac)*dataset.num_datapoints)
    n_test_steps = N_test//batch_size
    
    print(n_test_steps)
    
    with torch.no_grad():
        for step in range(n_test_steps):
            print("Step: " + str(step))
            batch = dataset[step*batch_size + N_train : step*batch_size+batch_size + N_train]
            im1_batch = Variable(torch.from_numpy(batch["depth_image1"]).float()).to(device)
            im2_batch = Variable(torch.from_numpy(batch["depth_image2"]).float()).to(device)
            transform_batch = Variable(torch.from_numpy(batch["transform"]).float()).to(device)
            pred_transform = model(im1_batch, im2_batch)
            loss = criterion(pred_transform, transform_batch)
            test_loss += loss.item()
        
    return test_loss/n_test_steps

if __name__ == '__main__':
    run_train = True
    train_frac = 0.8
    batch_size = 128
    dataset = TensorDataset.open("/nfs/diskstation/projects/rigid_body/")
    transform_pred_dim = 3
    im_shape = dataset[0]["depth_image1"].shape[:-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.0005)
    num_epochs = 100 
    train_losses = []
    test_losses = []
    losses_f_name = "results/losses.p"
    loss_plot_f_name = "plots/losses.png"
    
    if run_train:
        for epoch in range(num_epochs):
            train_loss = train(epoch, dataset)
            test_loss = test(epoch, dataset)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print("TRAIN LOSS")
            print(train_loss)
            print("TEST LOSS")
            print(test_loss)
            pickle.dump({"train_loss" : train_losses, "test_loss" : test_losses}, open( losses_f_name, "wb"))
            torch.save(model.state_dict(), "models/rb_net.pt")
            
    else:
        losses = pickle.load( open( losses_f_name, "rb" ) )
        train_returns = np.array(losses["train_loss"])
        test_returns = np.array(losses["test_loss"])
        print(train_returns)
        print(test_returns)
        plt.plot(np.arange(len(train_returns)) + 1, train_returns, label="Training Loss")
        plt.plot(np.arange(len(test_returns)) + 1, test_returns, label="Testing Loss")
        plt.xlabel("Training Iteration")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.legend(loc='best')
        plt.savefig(loss_plot_f_name)
        plt.close()

        
