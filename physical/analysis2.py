# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def Plot_Iteration(obj, iteration, file_end):
    losses = np.load("physical/objects/obj" + str(obj) + "/iteration" + str(iteration) + "/losses.npy")
    losses = np.round(losses,2)
    indices = np.arange(0,len(losses)+1, (len(losses)) // 4).astype(int)
    fig, axes = plt.subplots(1, 6, figsize=(18,3), constrained_layout = True)
    for j, ax in enumerate(axes.flatten()[:-1]):
        img_index = indices[j]
        try:
            img = plt.imread("physical/objects/obj" + str(obj) + "/iteration" + str(iteration) + "/depth_4/" + str(img_index)+ ".png")
        except:
            img = plt.imread("physical/objects/obj" + str(obj) + "/iteration" + str(iteration) + "/depth_4/" + str(img_index).zfill(2) + ".png")
        img2 = ax.imshow(img, vmin = np.min(img[img!=0])*0.9)
        # ax.set_aspect('auto')
        ax.axis('off')
        ax.set_xlim(125,525)
        ax.set_ylim(50,450)
        try:
            ax.set_title(unicode(str(losses[img_index]))+ u"°", fontsize=20)
        except:
            ax.set_title(unicode(str(losses[-1]))+ u"°", fontsize=20)
            
    ax = axes.flatten()[-1]
    img = plt.imread("physical/objects/obj" + str(obj) + "/iteration" + str(iteration) + "/depth_4/goal.png")
    img2 = ax.imshow(img, vmin = np.min(img[img!=0])*0.9)
    ax.set_xlim(125,525)
    ax.set_ylim(50,450)
    ax.axis('off')

    # plt.tight_layout()
    plt.savefig("physical/images/obj" + str(obj) + file_end)
    plt.close()

def Plot_Segmentation(obj, iteration):

    plt.close('all')
    fig = plt.figure(figsize=(18,6), constrained_layout=True)

    ax1 = plt.subplot(241)
    img = plt.imread("physical/objects/obj" + str(obj) + "/iteration" + str(iteration) + "/rgb_1/goal.png")
    img2 = ax1.imshow(img, vmin = np.min(img[img!=0])*0.9)
    ax1.set_xlim(125,525)
    ax1.set_ylim(60,420)
    ax1.axis('off')
    ax1.set_ylabel("RGB")

    ax2 = plt.subplot(245)
    img = plt.imread("physical/objects/obj" + str(obj) + "/iteration" + str(iteration) + "/depth_1/goal.png")
    img2 = ax2.imshow(img, vmin = np.min(img[img!=0])*0.9)
    ax2.set_xlim(125,525)
    ax2.set_ylim(60,420)
    ax2.axis('off')
    ax2.set_ylabel("Depth")

    ax3 = plt.subplot(242)
    img = plt.imread("physical/objects/obj" + str(obj) + "/iteration" + str(iteration) + "/rgb_2/goal.png")
    img2 = ax3.imshow(img, vmin = np.min(img[img!=0])*0.9)
    ax3.set_xlim(125,525)
    ax3.set_ylim(60,420)
    ax3.axis('off')
    
    ax4 = plt.subplot(246)
    img = plt.imread("physical/objects/obj" + str(obj) + "/iteration" + str(iteration) + "/depth_2/goal.png")
    img2 = ax4.imshow(img, vmin = np.min(img[img!=0])*0.9)
    ax4.set_xlim(125,525)
    ax4.set_ylim(60,420)
    ax4.axis('off')
    
    ax5 = plt.subplot(243)
    img = plt.imread("physical/objects/obj" + str(obj) + "/iteration" + str(iteration) + "/rgb_3/goal.png")
    img2 = ax5.imshow(img, vmin = np.min(img[img!=0])*0.9)
    ax5.set_xlim(125,525)
    ax5.set_ylim(60,420)
    ax5.axis('off')
    
    ax6 = plt.subplot(247)
    img = plt.imread("physical/objects/obj" + str(obj) + "/iteration" + str(iteration) + "/depth_3/goal.png")
    img2 = ax6.imshow(img, vmin = np.min(img[img!=0])*0.9)
    ax6.set_xlim(125,525)
    ax6.set_ylim(60,420)
    ax6.axis('off')

    ax7 = plt.subplot(144)
    img = plt.imread("physical/objects/obj" + str(obj) + "/iteration" + str(iteration) + "/depth_4/goal.png")
    img2 = ax7.imshow(img, vmin = np.min(img[img!=0])*0.9)
    ax7.set_xlim(150,510)
    ax7.set_ylim(60,420)
    ax7.axis('off')

    plt.savefig("physical/images/obj" + str(obj) + "_segmentation.png")
    # plt.show()


if __name__ == "__main__":
    colors = ["darkcyan", "magenta", "black", "brown", "skyblue"]
    objects = ["Dark Blue Pipe Connector", "Purple Clamp", "Black Tube", "Brown Rock Climbing Hold", "Sky Blue Bar Clamp"]

    for i in range(5):
        c = colors[i]
        final_loss = []
        for j in range(1,11):
            losses = np.load("physical/objects/obj" + str(i) + "/iteration" + str(j) + "/losses.npy")
            final_loss.append(losses[-1])
        Plot_Iteration(i,np.argmin(final_loss) + 1, "evo_best.png")
        Plot_Iteration(i,np.argmax(final_loss) + 1, "evo_worst.png")
        Plot_Segmentation(i,np.argmin(final_loss) + 1)
