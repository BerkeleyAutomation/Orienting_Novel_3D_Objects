import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':

    YAXIS = 'angle'
    # YAXIS = 'sm'

    bins = 6
    width = 0.25

    def loss2angle(loss):
        return np.arccos(1-loss) * 180 / np.pi * 2
        
    def process_histdata(histdata, yaxis = "sm", width=0):
        rotation_angles = [[] for i in range(bins)]
        if yaxis == "angle":
            for angle,loss,sm_loss in histdata:
                bin_num = np.min((int(angle // 5), bins-1))
                rotation_angles[bin_num].append(loss)

            mean_losses = [loss2angle(np.mean(ra)) for ra in rotation_angles]
            errors = [loss2angle(np.std(ra)) for ra in rotation_angles]
            x_loc = np.arange(bins) + width
        else:
            for angle,loss,sm_loss in histdata:
                bin_num = np.min((int(angle // 5), bins-1))
                rotation_angles[bin_num].append(sm_loss)

            mean_losses = [np.mean(ra) for ra in rotation_angles]
            errors = [np.std(ra) for ra in rotation_angles]
            x_loc = np.arange(bins) + width
        return mean_losses, errors, x_loc

    if YAXIS == "angle":
        histdata_cos = np.loadtxt("results/best_scoresv6/cos_blk1_emb1024_reg9_drop4_histdata.txt")
        histdata_hybrid = np.loadtxt("results/best_scoresv6/cos_sm_blk1_emb1024_reg9_drop4_histdata.txt")
        histdata_icp = np.loadtxt("results/best_scoresv6/icp_histdata.txt")
        mean_losses_hybrid, errors_hybrid, x_loc_hybrid = process_histdata(histdata_hybrid, "angle", 0)
        mean_losses_cos, errors_cos, x_loc_cos = process_histdata(histdata_cos, "angle", width)
        mean_losses_icp, errors_icp, x_loc_icp = process_histdata(histdata_icp, "angle", width*2)

        labels = [str(i) + "-" + str(i+5) for i in range(0,30,5)]
        plt.figure(figsize=(10,5))
        plt.bar(x_loc_hybrid, mean_losses_hybrid, yerr = errors_hybrid, width=width, label = "Hybrid Loss")
        plt.bar(x_loc_cos, mean_losses_cos, yerr = errors_cos, width=width, label = "Mean Loss")
        plt.bar(x_loc_icp, mean_losses_icp, yerr = errors_icp, width=width, label = "ICP")
        # plt.axhline(mean_loss, c = 'r')
        plt.xlabel("Rotation Angle (Degrees)", fontsize=17)
        plt.ylabel("Mean Angle Error (Degrees)", fontsize=17)
        # plt.ylim(0.0, (np.max(mean_losses)+np.max(errors))*1.1)
        plt.xticks([r + width*3/2 for r in range(6)], labels)
        plt.title("Mean Angle Error vs Rotation Angle on Non-Symmetric Dataset", fontsize=17)
        plt.legend(loc="best")
        plt.ylim(0,45)
        plt.savefig("plots/angle_loss.png")
        plt.close()

    else:
        histdata_cos = np.loadtxt("results/872obj/cos_blk1_emb1024_reg9_drop4_histdata.txt")
        histdata_hybrid = np.loadtxt("results/872obj/cos_sm_blk1_emb1024_reg9_drop4_histdata.txt")
        histdata_icp = np.loadtxt("results/872obj/icp_histdata.txt")
        mean_losses_hybrid, errors_hybrid, x_loc_hybrid = process_histdata(histdata_hybrid, "sm", 0)
        mean_losses_cos, errors_cos, x_loc_cos = process_histdata(histdata_cos, "sm", width)
        mean_losses_icp, errors_icp, x_loc_icp = process_histdata(histdata_icp, "sm", width*2)

        labels = [str(i) + "-" + str(i+5) for i in range(0,30,5)]
        plt.figure(figsize=(10,5))
        plt.bar(x_loc_hybrid, mean_losses_hybrid, yerr = errors_hybrid, width=width, label = "Hybrid Loss")
        plt.bar(x_loc_cos, mean_losses_cos, yerr = errors_cos, width=width, label = "Mean Loss")
        plt.bar(x_loc_icp, mean_losses_icp, yerr = errors_icp, width=width, label = "ICP")
        # plt.axhline(mean_loss, c = 'r')
        plt.xlabel("Rotation Angle (Degrees)", fontsize=17)
        plt.ylabel("Shape-Match Loss", fontsize=17)
        # plt.ylim(0.0, (np.max(mean_losses)+np.max(errors))*1.1)
        plt.xticks([r + width*3/2 for r in range(6)], labels)
        plt.title("Shape-Match Loss vs Rotation Angle on Full Dataset", fontsize=17)
        plt.legend(loc="best")
        plt.ylim(0,0.6)
        plt.savefig("plots/sm_loss.png")
        plt.close()


    # losses_546_hybrid = pickle.load(open("results/546objv3_cos_sm.p", "rb"))
    # losses_546_cosine = pickle.load(open("results/546objv3_cos.p", "rb"))
    # losses_82_hybrid = pickle.load(open("results/best_scoresv5_cos_sm.p", "rb"))
    # losses_hybrid = pickle.load(open("results/best_scoresv5_cos.p", "rb"))

    # test_returns_546_hybrid = np.array(losses_546_hybrid["test_loss"])
    # test_returns_546_cosine = np.array(losses_546_cosine["test_loss"])
    # test_returns_82_hybrid = np.array(losses_82_hybrid["test_loss"])
    # test_returns_hybrid = np.array(losses_hybrid["test_loss"])
    
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
    # plt.plot(np.arange(len(test_returns_hybrid)) + 1, test_returns_hybrid, label="Mean Loss")
    # plt.xlabel("Training Iteration")
    # plt.ylabel("Shape-Match Loss")
    # plt.title("Test Loss vs Training Iteration on Non-Symmetric Dataset")
    # plt.legend(loc='best')
    # plt.savefig("plots/test_curve_82.png")
    # plt.close()
