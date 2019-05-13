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

from autolab_core import YamlConfig
import torch.utils.data as data_utils
from unsupervised_rbt import TensorDataset
from unsupervised_rbt.models import ResNetSiameseNetwork, InceptionSiameseNetwork, ContextSiameseNetwork, ResNetObjIdPred
from unsupervised_rbt.models import LinearEmbeddingClassifier, Inc_LinearEmbeddingClassifier

# determine cuda
device = torch.device('cuda')
# turn of X-backend for matplotlib
os.system("echo \"backend: Agg\" > ~/.config/matplotlib/matplotlibrc")  

plot_lable = '_50obj_rand-init_1000'

def main(args):
    config = YamlConfig(args.config)
    
    # initialize model
    if args.model == 'ResNet':
        print(colored('------------- ResNet -------------', 'red'))
        model = LinearEmbeddingClassifier(config, num_classes=args.num_obj, dropout= args.dropout, init=args.init).to(device)
    else:
        print(colored('------------- Inception -------------', 'red'))
        model = Inc_LinearEmbeddingClassifier(config, num_classes=args.num_obj, dropout= args.dropout, init=args.init).to(device)
        
    if args.last_layers:
        print(colored('------------- Only last layers -------------', 'red'))
        # only train the fc layers on top
        optimizer_ft = optim.Adam([
                {'params': model.fc_1.parameters()},
                {'params': model.fc_2.parameters()},
                {'params': model.final_fc.parameters()},
            ], lr=args.lr)
    else: # train all
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
            with open('training_metadata/obj-pred_'+args.model + '.pkl', 'wb') as handle:
                    pickle.dump(to_pickle, handle)
            # save model
            torch.save(model.state_dict(), './trained_models/obj-pred_'+args.model+plot_lable+'.pkl')
            print('Model save to ./trained_models/'+args.model+'_'+args.dataset+plot_lable+'.pkl')
            # safe loss plots
            save_plots(train_losses, test_losses, train_accs, test_accs)
            
    
def train_model(model, criterion, optimizer, dataset, batch_size):
    
    for phase in ['train','val']:
        running_loss, correct, total = 0, 0, 0
        
        if phase == 'train':
            indices = dataset.split('train')[phase=='val'][:1000]
#             print("train with: ",len(indices))
        else:
            indices = dataset.split('train')[phase=='val'][:1000]
#             print("test with: ",len(indices))
            
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
                id_batch = Variable(torch.from_numpy(batch["obj_id"].astype(int))).to(device)
                
                # feed through model
                pred_id_1 = model(im1_batch)
                pred_id_2 = model(im2_batch)

                pred_id = torch.cat((pred_id_1, pred_id_2), 0)
                id_batch_full = torch.cat((id_batch, id_batch), 0)
                
                # evaluate
                _, predicted = torch.max(pred_id, 1)
                correct += (predicted == id_batch_full).sum().item()
                total += id_batch_full.size(0)
                loss = criterion(pred_id, id_batch_full)
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
    mpl.pyplot.savefig('./plots/'+'obj_pred'+args.model+'_'+args.dataset +'_'+ plot_lable)
    mpl.pyplot.close()

def parse_args():
    parser = argparse.ArgumentParser()
    
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/embedding_obj_prediction.yaml')
    parser.add_argument('-config', type=str, default=default_config_filename)
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-model', type=str, default='ResNet', choices=['ResNet','Inception','ContextPred'])
    parser.add_argument('-epochs', type=int, default=101)
    parser.add_argument('-dropout', type=bool, default=True)
    parser.add_argument('-init', type=bool, default=False)
    parser.add_argument('-num_obj', type=int, default=50)
    parser.add_argument('-last_layers', type=bool, default=False)
    args = parser.parse_args()
    parser.add_argument('-dataset_full', type=str, required=False) # never enter it, it's just to fill the whole path
    args.dataset_full = os.path.join('/raid/mariuswiggert', args.dataset)
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)