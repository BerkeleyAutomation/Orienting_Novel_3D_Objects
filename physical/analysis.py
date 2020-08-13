import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # test_546 = np.loadtxt("cfg/tools/train_split_546")
    # train_82 = np.loadtxt("cfg/tools/test_split_82")
    # test_82 = np.loadtxt("cfg/tools/train_split_82")

    # for i in test_546:
    #     if i in train_82 or i in test_82:
    #         print(int(i))

    # colors = ["skyblue", "orange", "olive", "red", "green"]
    colors = ["darkblue", "magenta", "black", "brown", "skyblue"]
    objects = ["Dark Blue Pipe Connector", "Purple Clamp", "Black Tube", "Brown Rock Climbing Hold", "SkyBlue Bar Clamp"]
    final_loss = []
    plt.figure(figsize=(10,5))
    num_objects = 5
    for i in range(num_objects):
        c = colors[i]
        for j in range(1,11):
            losses = np.load("physical/objects/obj" + str(i) + "/iteration" + str(j) + "/losses.npy")
            losses = np.pad(losses, (0,50-len(losses)), 'edge')
            plt.plot(losses, color = c)
            # plt.plot(l)
            final_loss.append(losses[-1])
    plt.title("Physical Controller Angle Error Evolution", fontsize=18)
    plt.ylabel("Angle Error (Degrees)", fontsize=17)
    plt.xlabel("Iteration Number", fontsize=17)
    plt.ylim((0,30))
    plt.legend(objects[:num_objects], loc = "upper right")
    plt.tight_layout()
    ax = plt.gca()
    leg = ax.get_legend()
    for i in range(num_objects):
        leg.legendHandles[i].set_color(colors[i])
    plt.savefig("physical/images/physical_controller_evolution_3.png")
    plt.close()

    plt.hist(final_loss, bins=np.arange(0,30))
    mean_angle_error = np.round(np.mean(final_loss), 2)
    median_angle_error = np.round(np.median(final_loss), 2)
    print("Median Angle Error", median_angle_error)
    plt.axvline(median_angle_error, color = "red", label = "Median Angle Error : {}".format(median_angle_error))
    plt.title("Controller Angle Error", fontsize=17)
    plt.xlabel("Angle Error (Degrees)", fontsize=17)
    plt.ylabel("Frequency", fontsize=17)
    plt.savefig("physical/loss_hist.png")
    less_than_5 = np.sum(np.array(final_loss) <= 5)
    print("Num less than 5: ", less_than_5, "out of", len(final_loss), "Percent:", less_than_5/len(final_loss)* 100)
    print("Num greater than 30: ", np.sum(np.array(final_loss) > 30))
