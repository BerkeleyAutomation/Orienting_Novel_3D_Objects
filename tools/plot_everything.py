import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    losses_546_hybrid = pickle.load(open("results/546objv3_cos_sm.p", "rb"))
    losses_546_cosine = pickle.load(open("results/546objv3_cos.p", "rb"))
    losses_82_hybrid = pickle.load(open("results/best_scoresv5_cos_sm.p", "rb"))
    losses_82_cosine = pickle.load(open("results/best_scoresv5_cos.p", "rb"))

    test_returns_546_hybrid = np.array(losses_546_hybrid["test_loss"])
    test_returns_546_cosine = np.array(losses_546_cosine["test_loss"])
    test_returns_82_hybrid = np.array(losses_82_hybrid["test_loss"])
    test_returns_82_cosine = np.array(losses_82_cosine["test_loss"])
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(test_returns_546_hybrid)) + 1, test_returns_546_hybrid, label="Hybrid Loss")
    plt.plot(np.arange(len(test_returns_546_cosine)) + 1, test_returns_546_cosine, label="Mean Loss")
    plt.xlabel("Training Iteration")
    plt.ylabel("Shape-Match Loss")
    plt.title("Test Loss vs Training Iteration on Full Dataset")
    plt.legend(loc='best')
    plt.savefig("plots/test_curve_546.png")
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(test_returns_82_hybrid)) + 1, test_returns_82_hybrid, label="Hybrid Loss")
    plt.plot(np.arange(len(test_returns_82_cosine)) + 1, test_returns_82_cosine, label="Mean Loss")
    plt.xlabel("Training Iteration")
    plt.ylabel("Shape-Match Loss")
    plt.title("Test Loss vs Training Iteration on Non-Symmetric Dataset")
    plt.legend(loc='best')
    plt.savefig("plots/test_curve_82.png")
    plt.close()

    ### CHANGE
    # angle_loss_546_hybrid = np.loadtxt("results/546objv3_cos_sm_histdata.txt") 
    # angle_loss_546_cosine = np.loadtxt("results/546objv3_cos_histdata.txt")
    # angle_loss_82_hybrid = np.loadtxt("results/best_scoresv5_cos_sm_histdata.txt")
    # angle_loss_82_cosine = np.loadtxt("results/best_scoresv5_cos_histdata.txt")

    bins = 6
    width = 0.25

    # 546 Hybrid
    rotation_angles_546_hybrid = [[] for i in range(bins)]
    for angle,loss,loss2 in angle_loss_546_hybrid:
        bin_num = np.min((int(angle // 5), bins-1))
        rotation_angles_546_hybrid[bin_num].append(loss2)

    mean_losses_546_hybrid = [np.mean(ra) for ra in rotation_angles_546_hybrid]
    errors_546_hybrid = [np.std(ra) for ra in rotation_angles_546_hybrid]
    rotation_angles_546_hybrid = [[] for i in range(bins)]
    x_loc_546_hybrid = np.arange(bins)

    # 546 Cosine
    rotation_angles_546_cosine = [[] for i in range(bins)]
    for angle,loss,loss2 in angle_loss_546_cosine:
        bin_num = np.min((int(angle // 5), bins-1))
        rotation_angles_546_cosine[bin_num].append(loss2)

    mean_losses_546_cosine = [np.mean(ra) for ra in rotation_angles_546_cosine]
    errors_546_cosine = [np.std(ra) for ra in rotation_angles_546_cosine]
    rotation_angles_546_cosine = [[] for i in range(bins)]
    x_loc_546_cosine = np.arange(bins) + width
    
    labels = [str(i) + "-" + str(i+5) for i in range(0,30,5)]
    plt.figure(figsize=(10,5))
    plt.bar(x_loc_546_hybrid, mean_losses_546_hybrid, yerr = errors_546_hybrid, width=width, label = "Hybrid Loss")
    plt.bar(x_loc_546_cosine, mean_losses_546_cosine, yerr = errors_546_cosine, width=width, label = "Mean Loss")
    # plt.axhline(mean_loss, c = 'r')
    plt.xlabel("Rotation Angle (Degrees)")
    plt.ylabel("Shape-Match Loss")
    # plt.ylim(0.0, (np.max(mean_losses)+np.max(errors))*1.1)
    plt.xticks([r + width/2 for r in range(6)], labels)
    plt.title("Shape-Match Loss vs Rotation Angle on Full Dataset")
    plt.legend(loc = "best")
    plt.savefig("plots/angle_loss_546.png")
    plt.close()

    def loss2angle(loss):
        return np.arccos(1-loss) * 180 / np.pi * 2
        
    # 82 Hybrid
    rotation_angles_82_hybrid = [[] for i in range(bins)]    
    for angle,loss,loss2 in angle_loss_82_hybrid:
        bin_num = np.min((int(angle // 5), bins-1))
        rotation_angles_82_hybrid[bin_num].append(loss)

    mean_losses_82_hybrid = [loss2angle(np.mean(ra)) for ra in rotation_angles_82_hybrid]
    errors_82_hybrid = [loss2angle(np.std(ra)) for ra in rotation_angles_82_hybrid]
    rotation_angles_82_hybrid = [[] for i in range(bins)]
    x_loc_82_hybrid = np.arange(bins)

    # 82 Cosine
    rotation_angles_82_cosine = [[] for i in range(bins)]    
    for angle,loss,loss2 in angle_loss_82_cosine:
        bin_num = np.min((int(angle // 5), bins-1))
        rotation_angles_82_cosine[bin_num].append(loss)

    mean_losses_82_cosine = [loss2angle(np.mean(ra)) for ra in rotation_angles_82_cosine]
    errors_82_cosine = [loss2angle(np.std(ra)) for ra in rotation_angles_82_cosine]
    rotation_angles_82_cosine = [[] for i in range(bins)]
    x_loc_82_cosine = np.arange(bins) + width

    labels = [str(i) + "-" + str(i+5) for i in range(0,30,5)]
    plt.figure(figsize=(10,5))
    plt.bar(x_loc_82_hybrid, mean_losses_82_hybrid, yerr = errors_82_hybrid, width=width, label = "Hybrid Loss")
    plt.bar(x_loc_82_cosine, mean_losses_82_cosine, yerr = errors_82_cosine, width=width, label = "Mean Loss")
    # plt.axhline(mean_loss, c = 'r')
    plt.xlabel("Rotation Angle (Degrees)")
    plt.ylabel("Mean Angle Error (Degrees)")
    # plt.ylim(0.0, (np.max(mean_losses)+np.max(errors))*1.1)
    plt.xticks([r + width/2 for r in range(6)], labels)
    plt.title("Mean Angle Error vs Rotation Angle on Non-Symmetric Dataset")
    plt.legend(loc="best")
    plt.savefig("plots/angle_loss_82.png")
    plt.close()

