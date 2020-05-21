import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':

    YAXIS = 'sm'

    ### CHANGE
    histdata_1 = np.loadtxt("results/546objv3/cos_sm_blk4_emb1024_reg7_drop4_histdata.txt") 
    histdata_2 = np.loadtxt("results/546objv4/cos_sm_blk1_emb1024_reg7_drop6_histdata.txt")

    bins = 6
    width = 0.25

    def loss2angle(loss):
        return np.arccos(1-loss) * 180 / np.pi * 2
        
    if YAXIS == "angle":
        rotation_angles_1 = [[] for i in range(bins)]    
        for angle,loss,loss2 in histdata_1:
            bin_num = np.min((int(angle // 5), bins-1))
            rotation_angles_1[bin_num].append(loss)

        mean_losses_1 = [loss2angle(np.mean(ra)) for ra in rotation_angles_1]
        errors_1 = [loss2angle(np.std(ra)) for ra in rotation_angles_1]
        rotation_angles_1 = [[] for i in range(bins)]
        x_loc_1 = np.arange(bins)

        rotation_angles_2 = [[] for i in range(bins)]    
        for angle,loss,loss2 in histdata_2:
            bin_num = np.min((int(angle // 5), bins-1))
            rotation_angles_2[bin_num].append(loss)

        mean_losses_2 = [loss2angle(np.mean(ra)) for ra in rotation_angles_2]
        errors_2 = [loss2angle(np.std(ra)) for ra in rotation_angles_2]
        rotation_angles_2 = [[] for i in range(bins)]
        x_loc_2 = np.arange(bins) + width

        labels = [str(i) + "-" + str(i+5) for i in range(0,30,5)]
        plt.figure(figsize=(10,5))
        plt.bar(x_loc_1, mean_losses_1, yerr = errors_1, width=width, label = "Hybrid Loss")
        plt.bar(x_loc_2, mean_losses_2, yerr = errors_2, width=width, label = "Mean Loss")
        # plt.axhline(mean_loss, c = 'r')
        plt.xlabel("Rotation Angle (Degrees)")
        plt.ylabel("Mean Angle Error (Degrees)")
        # plt.ylim(0.0, (np.max(mean_losses)+np.max(errors))*1.1)
        plt.xticks([r + width/2 for r in range(6)], labels)
        plt.title("Mean Angle Error vs Rotation Angle on Full Dataset")
        plt.legend(loc="best")
        plt.savefig("plots/angle_loss.png")
        plt.close()

    else:
        rotation_angles_1 = [[] for i in range(bins)]
        for angle,loss,loss2 in histdata_1:
            bin_num = np.min((int(angle // 5), bins-1))
            rotation_angles_1[bin_num].append(loss2)

        mean_losses_1 = [np.mean(ra) for ra in rotation_angles_1]
        errors_1 = [np.std(ra) for ra in rotation_angles_1]
        rotation_angles_1 = [[] for i in range(bins)]
        x_loc_1 = np.arange(bins)

        # 546 Cosine
        rotation_angles_2 = [[] for i in range(bins)]
        for angle,loss,loss2 in histdata_2:
            bin_num = np.min((int(angle // 5), bins-1))
            rotation_angles_2[bin_num].append(loss2)

        mean_losses_2 = [np.mean(ra) for ra in rotation_angles_2]
        errors_2 = [np.std(ra) for ra in rotation_angles_2]
        rotation_angles_2 = [[] for i in range(bins)]
        x_loc_2 = np.arange(bins) + width
        
        labels = [str(i) + "-" + str(i+5) for i in range(0,30,5)]
        plt.figure(figsize=(10,5))
        plt.bar(x_loc_1, mean_losses_1, yerr = errors_1, width=width, label = "No DR")
        plt.bar(x_loc_2, mean_losses_2, yerr = errors_2, width=width, label = "DR on Center of Mass")
        # plt.axhline(mean_loss, c = 'r')
        plt.xlabel("Rotation Angle (Degrees)")
        plt.ylabel("Shape-Match Loss")
        # plt.ylim(0.0, (np.max(mean_losses)+np.max(errors))*1.1)
        plt.xticks([r + width/2 for r in range(6)], labels)
        plt.title("Shape-Match Loss vs Rotation Angle on Full Dataset")
        plt.legend(loc = "best")
        plt.savefig("plots/sm_loss.png")
        plt.close()


    # losses_546_hybrid = pickle.load(open("results/546objv3_cos_sm.p", "rb"))
    # losses_546_cosine = pickle.load(open("results/546objv3_cos.p", "rb"))
    # losses_82_hybrid = pickle.load(open("results/best_scoresv5_cos_sm.p", "rb"))
    # losses_2 = pickle.load(open("results/best_scoresv5_cos.p", "rb"))

    # test_returns_546_hybrid = np.array(losses_546_hybrid["test_loss"])
    # test_returns_546_cosine = np.array(losses_546_cosine["test_loss"])
    # test_returns_82_hybrid = np.array(losses_82_hybrid["test_loss"])
    # test_returns_2 = np.array(losses_2["test_loss"])
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(np.arange(len(test_returns_546_hybrid)) + 1, test_returns_546_hybrid, label="Hybrid Loss")
    # plt.plot(np.arange(len(test_returns_546_cosine)) + 1, test_returns_546_cosine, label="Mean Loss")
    # plt.xlabel("Training Iteration")
    # plt.ylabel("Shape-Match Loss")
    # plt.title("Test Loss vs Training Iteration on Full Dataset")
    # plt.legend(loc='best')
    # plt.savefig("plots/test_curve_546.png")
    # plt.close()
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(np.arange(len(test_returns_82_hybrid)) + 1, test_returns_82_hybrid, label="Hybrid Loss")
    # plt.plot(np.arange(len(test_returns_2)) + 1, test_returns_2, label="Mean Loss")
    # plt.xlabel("Training Iteration")
    # plt.ylabel("Shape-Match Loss")
    # plt.title("Test Loss vs Training Iteration on Non-Symmetric Dataset")
    # plt.legend(loc='best')
    # plt.savefig("plots/test_curve_82.png")
    # plt.close()
