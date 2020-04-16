import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    losses = np.load("controller/results/losses.npy")
    test_546 = np.loadtxt("cfg/tools/train_split_546")
    train_82 = np.loadtxt("cfg/tools/test_split_82")
    test_82 = np.loadtxt("cfg/tools/train_split_82")

    # for i in test_546:
    #     if i in train_82 or i in test_82:
    #         print(int(i))


    # colors = ["skyblue", "orange", "olive", "red", "green"]
    final_loss = []
    for i,obj in enumerate(losses):
        # c = colors[i]
        for l in obj:
            # plt.plot(l, color = c)
            plt.plot(l)
            final_loss.append(l[-1])
    plt.title("Controller Angle Error Evolution")
    plt.ylabel("Angle Difference (Degrees)")
    plt.xlabel("Iteration Number")
    plt.savefig("controller/results/loss.png")
    plt.xlim((0,losses.shape[1]))
    plt.ylim((0,30))
    plt.close()

    plt.hist(final_loss, bins=np.arange(0,30))
    mean_angle_error = np.round(np.mean(final_loss), 2)
    median_angle_error = np.round(np.median(final_loss), 2)
    print("Mean Angle Error", mean_angle_error)
    plt.axvline(median_angle_error, color = "red", label = "Median Angle Error : {}".format(median_angle_error))
    plt.title("Controller Angle Error")
    plt.xlabel("Angle Difference (Degrees)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("controller/results/loss_hist.png")
    less_than_5 = np.sum(np.array(final_loss) <= 5)
    print("Num less than 5: ", less_than_5, "out of", len(final_loss), "Percent:", less_than_5/len(final_loss)* 100)
    print("Num greater than 30: ", np.sum(np.array(final_loss) > 30))
