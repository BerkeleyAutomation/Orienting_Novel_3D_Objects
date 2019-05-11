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
import matplotlib.pyplot as plt
from tqdm import tqdm

from autolab_core import YamlConfig, RigidTransform
from unsupervised_rbt import TensorDataset
from unsupervised_rbt.models import ResNetSiameseNetwork
from perception import DepthImage, RgbdImage

# TODO: Make this a real architecture, this is just a minimum working example for now
# TODO: Improve batching speed/data loading, its still kind of slow rn

def get_params_to_train(model):
    return model.parameters()
    params = []
    r = model.resnet
    layers = [r.layer3, r.layer4, r.linear, model.fc_1, model.fc_2, model.final_fc]
#     layers = [model.fc_1, model.fc_2, model.final_fc]
    for layer in layers:
        params.extend(layer.parameters())
    return params


def generate_data(dataset):
    im1s, im2s, labels = [], [], []
    for _ in range(1000):
        dp1_idx = np.random.randint(dataset.num_datapoints)
        dp2_idx, label = dp1_idx, 1 # same object
        
        im1_idx = np.random.randint(20)
        im2_idx = np.random.randint(20)
        
        im1s.append(255 * dataset[dp1_idx]['depth_images'][im1_idx])

        if np.random.random() < 0.5: # different object
            while dp2_idx == dp1_idx:
                dp2_idx = np.random.randint(dataset.num_datapoints)
            label = 0

        im2s.append(255 * dataset[dp2_idx]['depth_images'][im2_idx])
        labels.append(label)
    im1s, im2s, labels = np.array(im1s), np.array(im2s), np.array(labels)
    return np.expand_dims(im1s, 1), np.expand_dims(im2s, 1), labels

def train(im1s, im2s, labels, batch_size):
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    N_train = int(0.8 * im1s.shape[0])
    n_train_steps = N_train//batch_size
    for step in tqdm(range(n_train_steps)):
        im1_batch   = Variable(torch.from_numpy(im1s[step*batch_size : (step+1)*batch_size]).float()).to(device)
        im2_batch   = Variable(torch.from_numpy(im2s[step*batch_size : (step+1)*batch_size]).float()).to(device)
        label_batch = Variable(torch.from_numpy(labels[step*batch_size : (step+1)*batch_size]).float()).to(device)

#         for i in range(batch_size):
#             plt.subplot(121)
#             depth_image_show1 = im1s[step*batch_size + i][0]
#             plt.imshow(depth_image_show1, cmap='gray')
#             plt.subplot(122)
#             depth_image_show2 = im2s[step*batch_size + i][0]
#             plt.imshow(depth_image_show2, cmap='gray')
#             plt.title('Transform: {}'.format(labels[step*batch_size + i]))
#             plt.show()
       
        optimizer.zero_grad()
        prob = model(im1_batch, im2_batch)
        loss = criterion(prob, label_batch.long())
        _, predicted = torch.max(prob, 1)

        correct += (predicted == label_batch.long()).sum().item()
        total += label_batch.size(0)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    class_acc = 100 * correct/total
    return train_loss/n_train_steps, class_acc

def test(im1s, im2s, labels, batch_size):
    model.eval()
    test_loss, correct, total = 0, 0, 0

    N_test = int(0.2 * im1s.shape[0])
    N_train = int(0.8 * im1s.shape[0])
    n_test_steps = N_test // batch_size
    im1s, im2s, labels = im1s[N_train:], im2s[N_train:], labels[N_train:]
    with torch.no_grad():
        for step in tqdm(range(n_test_steps)):
            im1_batch   = Variable(torch.from_numpy(im1s[step*batch_size   : (step+1)*batch_size]).float()).to(device)
            im2_batch   = Variable(torch.from_numpy(im2s[step*batch_size   : (step+1)*batch_size]).float()).to(device)
            label_batch = Variable(torch.from_numpy(labels[step*batch_size : (step+1)*batch_size]).float()).to(device)
       
            optimizer.zero_grad()
            prob = model(im1_batch, im2_batch)
            loss = criterion(prob, label_batch.long())
            _, predicted = torch.max(prob, 1)
            correct += (predicted == label_batch.long()).sum().item()
            total += label_batch.size(0)
            
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
        imshow(torchvision.utils.make_grid(model.resnet.conv1.weight))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/semisup_rbt_train.yaml')
    parser.add_argument('-config', type=str, default=default_config_filename)
    parser.add_argument('-dataset', type=str, required=True)
#     parser.add_argument('-unsup_model', type=str, required=True)
    args = parser.parse_args()
    args.dataset = os.path.join('/nfs/diskstation/projects/unsupervised_rbt', args.dataset)
    return args

if __name__ == '__main__':
    args = parse_args()
    config = YamlConfig(args.config)

    if not args.test:
        dataset = TensorDataset.open(args.dataset)
        im1s, im2s, labels = generate_data(dataset)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetSiameseNetwork(config['pred_dim'], n_blocks=1, embed_dim=20).to(device)
#         model.load_state_dict(torch.load(config['unsup_model_path']))
        new_state_dict = model.state_dict()
        
        layers_to_keep = tuple(['resnet.layer1', 'resnet.layer2'])
        load_params = torch.load(config['unsup_model_path'])
#         layers_to_keep = tuple(load_params.keys())
#         print(load_params.keys())
#         assert(False)
        
        for layer_name in load_params:
            if not layer_name.startswith(layers_to_keep):
                del load_params[layer_name]
                
        new_state_dict.update(load_params)
        model.load_state_dict(new_state_dict)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params_to_train(model))
        
        if not os.path.exists(args.dataset + "/splits/train"):
            print("Created Train Split")
            dataset.make_split("train", train_pct=0.8)

        train_losses, test_losses, train_accs, test_accs = [], [], [], []
        for epoch in range(config['num_epochs']):
            train_loss, train_acc = train(im1s, im2s, labels, config['batch_size'])
            test_loss, test_acc = test(im1s, im2s, labels, config['batch_size'])
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print("Epoch %d, Train Loss = %f, Train Acc = %.2f %%, Test Loss = %f, Test Acc = %.2f %%" % (epoch, train_loss, train_acc, test_loss, test_acc))
            pickle.dump({"train_loss" : train_losses, "train_acc" : train_accs, "test_loss" : test_losses, "test_acc" : test_accs}, open( config['losses_f_name'], "wb"))
            torch.save(model.state_dict(), config['model_save_dir'])
            
    else:
#         model = ResNetSiameseNetwork(config['pred_dim']).to(device)
#         model.load_state_dict(torch.load(config['model_save_dir']))
#         display_conv_layers(model)

        losses = pickle.load( open( config['losses_f_name'], "rb" ) )
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
        plt.savefig(config['losses_plot_f_name'])
        plt.close()
        
        plt.plot(np.arange(len(train_accs)) + 1, train_accs, label="Training Acc")
        plt.plot(np.arange(len(test_accs)) + 1, test_accs, label="Testing Acc")
        plt.xlabel("Training Iteration")
        plt.ylabel("Classification Accuracy")
        plt.title("Training Curve")
        plt.legend(loc='best')
        plt.savefig(config['accs_plot_f_name'])
        plt.close()
