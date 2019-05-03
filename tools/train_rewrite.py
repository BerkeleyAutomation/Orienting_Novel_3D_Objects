import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import pickle
import matplotlib as mpl
from tqdm import tqdm
from termcolor import colored

import torch.utils.data as data_utils
from unsupervised_rbt import TensorDataset
from unsupervised_rbt.models import ResNetSiameseNetwork, InceptionSiameseNetwork, ContextSiameseNetwork

# determine cuda
device = torch.device('cuda')
# turn of X-backend for matplotlib
os.system("echo \"backend: Agg\" > ~/.config/matplotlib/matplotlibrc")  

def main(args):
    # initialize model
    if args.model == 'ResNet':
        model = ResNetSiameseNetwork(transform_pred_dim=4, dropout= args.dropout).to(device)
    elif args.model == 'Inception':
        model = InceptionSiameseNetwork(transform_pred_dim=4, dropout= args.dropout).to(device)
    elif args.model == 'ContextPred':
        model = ContextSiameseNetwork(transform_pred_dim=4).to(device)
    
    # setup optimizer and loss
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    
    # load dataset
    dataset = TensorDataset.open(args.dataset_full)
    if not os.path.exists(args.dataset_full + "/splits/train"):
        print("Created Train Split")
        dataset.make_split("train", train_pct=0.8)
    
    print(colored('------------- Start Training with datasize ' + str(len(dataset.split('train')[0]) + len(dataset.split('train')[1])) + ' -------------', 'green'))

    # Train and evaluate
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    for epoch in range(args.epochs):
        print(colored('------------- epoch ' + str(epoch) + ' -------------', 'red'))
        train_loss, train_acc, test_loss, test_acc = train_model(model, criterion, optimizer_ft, dataset, args.batch_size)
        
        # Log 
        print("Train Loss = %f, Train Acc = %f %%, Test Loss = %f, Test Acc = %f %%" % (train_loss, train_acc, test_loss, test_acc))
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # safe the losses and model
        if (epoch) % 10 == 0:
            to_pickle = {
                    'train_losses': train_losses,
                    'train_accs':train_accs,
                    'test_losses': test_losses,
                    'test_accs': test_accs
                }
            # safe losses
            with open('training_metadata/'+args.model+'_'+args.dataset+str(args.dropout)+'.pkl', 'wb') as handle:
                    pickle.dump(to_pickle, handle)
            # save model
            torch.save(model.state_dict(), './trained_models/'+args.model+'_'+args.dataset+str(args.dropout)+'.pkl')
            print('Model save to ./trained_models/'+args.model+'_'+args.dataset+'.pkl')
            # safe loss plots
            save_plots(train_losses, test_losses, train_accs, test_accs)
            
    
def train_model(model, criterion, optimizer, dataset, batch_size):
    
    for phase in ['train','val']:
        running_loss, correct, total = 0, 0, 0
        
        if phase == 'train':
            indices = dataset.split('train')[phase=='val']
        else:
            indices = dataset.split('train')[phase=='val']
            
        #indices = dataset.split('train')[phase=='val']
        N_train = len(indices)
        minibatches = np.full(N_train//batch_size, batch_size, dtype=np.int)
        minibatches[:N_train % batch_size] += 1 # correct if the folds can't be even sized

        # 1) Set-Up
        if phase == 'train':
            model.train()  # Set model to training mode
            torch.set_grad_enabled(True)
            
        else:
            model.eval()
            torch.set_grad_enabled(False) # we don't want to update any weights
            
        # 2) Iterate through the respective dataset
        with tqdm(total=N_train//batch_size) as progress_bar:
            for step, batch_size in enumerate(minibatches):
                # extract the data
                batch = dataset.get_item_list(indices[step*batch_size : (step+1)*batch_size])
                depth_image1 = batch["depth_image1"] * 255
                depth_image2 = batch["depth_image2"] * 255
    #             depth_image1 = (batch["depth_image1"] * 255).astype(int)
    #             depth_image2 = (batch["depth_image2"] * 255).astype(int)
                im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
                im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
                transform_batch = Variable(torch.from_numpy(batch["transform"].astype(int))).to(device)

                # feed through model
                pred_transform = model(im1_batch, im2_batch)

                # evaluate
                _, predicted = torch.max(pred_transform, 1)
                correct += (predicted == transform_batch).sum().item()
                total += transform_batch.size(0)
                loss = criterion(pred_transform, transform_batch)
                running_loss += loss.item()

                # gradient step
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                progress_bar.update()
        
        # track real loss
        if phase == 'train':
            train_acc = 100 * correct/total
            train_loss = running_loss/(step+1)
        else:
            test_acc = 100 * correct/total
            test_loss = running_loss/(step+1)
    
    return  train_loss, train_acc, test_loss, test_acc

def save_plots(train_losses, test_losses, train_accs, test_accs):
    # transform to np arrays
    train_returns = np.array(train_losses)
    test_returns = np.array(test_losses)
    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)
    
    mpl.pyplot.subplot(211)
    mpl.pyplot.plot(np.arange(len(train_returns)) + 1, train_returns, label="Training Loss")
    mpl.pyplot.plot(np.arange(len(test_returns)) + 1, test_returns, label="Testing Loss")
    mpl.pyplot.xlabel("Training Iteration")
    mpl.pyplot.ylabel("Loss")
    mpl.pyplot.title("Training Curve")
    mpl.pyplot.legend(loc='best')
    
    mpl.pyplot.subplot(212)
    mpl.pyplot.plot(np.arange(len(train_accs)) + 1, train_accs, label="Training Acc")
    mpl.pyplot.plot(np.arange(len(test_accs)) + 1, test_accs, label="Testing Acc")
    mpl.pyplot.xlabel("Training Iteration")
    mpl.pyplot.ylabel("Classification Accuracy")
    mpl.pyplot.title("Training Curve")
    mpl.pyplot.legend(loc='best')
    mpl.pyplot.savefig('./plots/'+'Train_'+args.model+'_'+args.dataset + str(args.dropout))
    mpl.pyplot.close()
#     mpl.pyplot.savefig('./plots/'+'Acc_'+args.model+'_'+args.dataset)
#     mpl.pyplot.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True, choices=['z-axis-only','z-axis-only-obj-pred', 'xyz-axis-obj-pred', 'xyz-axis'])
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-model', type=str, default='ResNet', choices=['ResNet','Inception','ContextPred'])
    parser.add_argument('-epochs', type=int, default=101)
    parser.add_argument('-dropout', type=bool, default=True)
    args = parser.parse_args()
    parser.add_argument('-dataset_full', type=str, required=False) # never enter it, it's just to fill the whole path
    args.dataset_full = os.path.join('/raid/mariuswiggert', args.dataset)
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)