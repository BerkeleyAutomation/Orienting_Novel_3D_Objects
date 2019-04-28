import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import itertools
import torch
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns

from autolab_core import YamlConfig, RigidTransform
from unsupervised_rbt import TensorDataset
from unsupervised_rbt.models import SiameseNetwork
from perception import DepthImage, RgbdImage

def test(dataset, batch_size):
    model.eval()
    test_loss, correct, total = 0, 0, 0

    test_indices = dataset.split('train')[1]
    N_test = len(test_indices)
    n_test_steps = N_test // batch_size
    outputs, labels = [], []
    with torch.no_grad():
        for step in tqdm(range(n_test_steps)):
            batch = dataset.get_item_list(test_indices[step*batch_size : (step+1)*batch_size])
            depth_image1 = (batch["depth_image1"] * 255).astype(int)
            depth_image2 = (batch["depth_image2"] * 255).astype(int)
            im1_batch = Variable(torch.from_numpy(depth_image1).float()).to(device)
            im2_batch = Variable(torch.from_numpy(depth_image2).float()).to(device)
            
            outputs.extend(model.resnet(im1_batch).cpu().data.numpy())
            outputs.extend(model.resnet(im2_batch).cpu().data.numpy())
            labels.extend(list(batch['obj_id']))
            labels.extend(list(batch['obj_id']))
    
    labels = np.array(labels)
    # tSNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(outputs)
    fashion_scatter(tsne_results, labels)
    
#     # KNN vis
#     # Pick 10 test images:
#     depth_image1
    

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = 51
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    
    plt.savefig('plots/tsne_vis')

    return f, ax, sc, txts

def parse_args():
    parser = argparse.ArgumentParser()
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/rb_net_train.yaml')
    parser.add_argument('-config', type=str, default=default_config_filename)
    parser.add_argument('-dataset', type=str)
    args = parser.parse_args()
    args.dataset = os.path.join('/nfs/diskstation/projects/unsupervised_rbt', args.dataset)
    return args

if __name__ == '__main__':
    args = parse_args()
    config = YamlConfig(args.config)

    dataset = TensorDataset.open(args.dataset)
    
    if not os.path.exists(args.dataset + "/splits/train"):
        print("Created Train Split")
        dataset.make_split("train", train_pct=0.8)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork(config['transform_pred_dim']).to(device)
    model.load_state_dict(torch.load(config['model_save_dir']))
    test(dataset, config['batch_size'])
