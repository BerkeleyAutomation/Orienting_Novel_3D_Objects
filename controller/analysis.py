import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # losses = np.load("controller/results/losses.npy")
    losses = np.load("controller/results/losses_FINAL_22x100.npy")

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
    print("Mean Angle Error", mean_angle_error)
    plt.axvline(median_angle_error, color = "red", label = "Median Angle Error : {}".format(median_angle_error))
    plt.title("Simulation Controller Final Angle Error", fontsize=16)
    plt.xlabel("Final Angle Error (Degrees)", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.legend()
    plt.savefig("controller/results/loss_hist.png")
    less_than_5 = np.sum(np.array(final_loss) <= 5)
    print("Num less than 5: ", less_than_5, "out of", len(final_loss), "Percent:", less_than_5/len(final_loss)* 100)
    print("Num greater than 30: ", np.sum(np.array(final_loss) > 30))
