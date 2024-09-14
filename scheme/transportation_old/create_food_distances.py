import os

import numpy as np
import matplotlib.pyplot as plt


def each_distance():
    cd = os.path.dirname(__file__)
    res_dir = os.path.join(cd, "results")

    for task_dir in os.listdir(res_dir):
        task_dir = os.path.join(res_dir, task_dir)

        if not os.path.isdir(task_dir):
            continue

        hist_path = os.path.join(task_dir, "food_distance.npy")
        if not os.path.exists(hist_path):
            continue

        food_distance = np.load(os.path.join(task_dir, "food_distance.npy"))
        xs = np.arange(0, food_distance.shape[0]) * 0.033333

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        for i in range(2):
            axis.plot(xs, food_distance[:, i])
        fig.savefig(os.path.join(task_dir, "food_distance.svg"))
        plt.close(fig)


def comprehensive_distance():
    cd = os.path.dirname(__file__)
    res_dir = os.path.join(cd, "results")

    dists: list[(str, np.ndarray)] = []

    for task_dir in os.listdir(res_dir):
        task_dir = os.path.join(res_dir, task_dir)

        if not os.path.isdir(task_dir):
            continue

        hist_path = os.path.join(task_dir, "food_distance.npy")
        if not os.path.exists(hist_path):
            continue

        dists.append(
            (os.path.basename(task_dir), np.load(os.path.join(task_dir, "food_distance.npy")))
        )

    xs = np.arange(0, dists[0][1].shape[0]) * 0.033333

    fig1 = plt.figure()
    fig2 = plt.figure()
    axis1 = fig1.add_subplot(1, 1, 1)
    axis2 = fig2.add_subplot(1, 1, 1)
    for i, (n, s) in enumerate(dists):
        print(i)
        axis1.plot(xs, s[:, 0], label=n)
        axis2.plot(xs, s[:, 1], label=n)
    axis1.legend()
    axis2.legend()
    fig1.savefig(os.path.join(cd, "results/food600.svg"))
    fig2.savefig(os.path.join(cd, "results/food1000.svg"))
    plt.close(fig1)
    plt.close(fig2)


def main():
    each_distance()
    comprehensive_distance()


if __name__ == '__main__':
    main()
