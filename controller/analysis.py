import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    losses = np.load("controller/results/losses_ecc3.npy")
    # losses = np.load("controller/results/losses_FINAL_22x100.npy")

    colors = ["magenta", "darkblue", "green", "red", "turquoise"]
    final_loss = []
    plt.figure(figsize=(10,6))
    for i,obj in enumerate(losses):
        # c = colors[i]
        for l in obj:
            # plt.plot(l, color = c)
            plt.plot(l)
            final_loss.append(l[-1])
    plt.title("Simulation Controller Angle Error Evolution", fontsize=20)
    plt.ylabel("Angle Error (Degrees)", fontsize=20)
    plt.xlabel("Iteration Number", fontsize=20)
    plt.xlim(0,20)
    plt.xticks(range(0,21,2))
    plt.ylim((0,30))
    plt.legend(("Mug", "Golem", "Pharaoh", "Cat"))
    plt.tight_layout()
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(colors[0])
    leg.legendHandles[1].set_color(colors[1])
    leg.legendHandles[2].set_color(colors[2])
    leg.legendHandles[3].set_color(colors[3])
    plt.savefig("controller/results/loss.png")
    plt.close()

    plt.hist(final_loss, bins=np.arange(0,30))
    mean_angle_error = np.round(np.mean(final_loss), 2)
    median_angle_error = np.round(np.median(final_loss), 2)
    print("Median Angle Error", median_angle_error)
    plt.axvline(median_angle_error, color = "red", label = "Median Angle Error : {}".format(median_angle_error))
    plt.title("Simulation Controller Final Angle Error", fontsize=16)
    plt.xlabel("Final Angle Error (Degrees)", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.legend()
    plt.savefig("controller/results/loss_hist.png")
    less_than_5 = np.sum(np.array(final_loss) <= 5)
    print("Num less than 5: ", less_than_5, "out of", len(final_loss), "Percent:", less_than_5/len(final_loss)* 100)
    print("Num greater than 30: ", np.sum(np.array(final_loss) > 30))
    plt.close()
    
    fits = np.load("controller/results/fits_ecc3.npy")

    colors = ["magenta", "darkblue", "green", "red", "turquoise"]
    final_fit = []
    plt.figure(figsize=(10,6))
    for i,obj in enumerate(fits):
        c = colors[i]
        for l in obj:
            plt.plot(l, color = c)
            # plt.plot(l)
            final_fit.append(l[-1])
    plt.title("Simulation Controller Percent Fit Evolution", fontsize=20)
    plt.ylabel("Percent Fit", fontsize=20)
    plt.xlabel("Iteration Number", fontsize=20)
    plt.xlim(0,5)
    plt.xticks(range(0,5+1,1))
    plt.ylim((0.3,1))
    plt.legend(("Raspberry Pi Case", "Endstop Holder", "Shield Part", "Industrial Part"))
    plt.tight_layout()
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(colors[0])
    leg.legendHandles[1].set_color(colors[1])
    leg.legendHandles[2].set_color(colors[2])
    leg.legendHandles[3].set_color(colors[3])
    plt.savefig("controller/results/fits.png")
    plt.close()

    plt.hist(final_fit, bins=np.arange(0.7,1,0.01))
    mean_fit = np.round(np.mean(final_fit), 2)
    median_fit = np.round(np.median(final_fit), 3)
    mean_fit = np.round(np.mean(final_fit), 3)
    print("Median Percent Fit", median_fit)
    print("Mean Percent Fit", mean_fit)
    plt.axvline(median_fit, color = "lime", label = "Median Final Percent Fit : {}".format(median_fit))
    plt.axvline(mean_fit, color = "red", label = "Mean Final Percent Fit : {}".format(mean_fit))
    plt.title("Simulation Controller Final Percent Fit", fontsize=16)
    plt.xlabel("Final Percent Fit", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.legend()
    plt.savefig("controller/results/fit_hist.png")
    # less_than_5 = np.sum(np.array(final_fit) <= 5)
    # print("Num less than 5: ", less_than_5, "out of", len(final_fit), "Percent:", less_than_5/len(final_fit)* 100)
    # print("Num greater than 30: ", np.sum(np.array(final_fit) > 30))
