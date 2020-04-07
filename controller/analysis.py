import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    losses = np.load("controller/results/losses.npy")

    colors = ["skyblue", "orange", "olive", "red", "green"]
    final_loss = []
    for i,obj in enumerate(losses):
        c = colors[i]
        for l in obj:
            plt.plot(l, color = c)
            final_loss.append(l[-1])
    plt.title("Controller Angle Error Evolution")
    plt.ylabel("Angle Difference (Degrees)")
    plt.xlabel("Iteration Number")
    plt.savefig("controller/results/loss.png")
    plt.xlim((0,losses.shape[1]))
    plt.ylim((0,30))
    plt.close()

    plt.hist(final_loss, bins=np.arange(0,26))
    plt.title("Controller Angle Error")
    plt.xlabel("Angle Difference (Degrees)")
    plt.ylabel("Frequency")
    plt.savefig("controller/results/loss_hist.png")
