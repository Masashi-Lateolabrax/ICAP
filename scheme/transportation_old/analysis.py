import os

import numpy as np
import matplotlib.pyplot as plt

from lib.optimizer import Hist

from environments.collect_feed_without_obstacle import RobotBrain


def plot(task_dir):
    features = np.load(os.path.join(task_dir, "features.npy"))
    outputs = np.load(os.path.join(task_dir, "outputs.npy"))

    xs = np.arange(0, features.shape[0]) * 0.033333

    for bi in range(9):
        fig = plt.figure()
        axis1 = fig.add_subplot(1, 1, 1)
        axis2 = axis1.twinx()
        axis1.plot(xs, features[:, bi, 0], color="green")
        axis1.plot(xs, features[:, bi, 3], color="blue")
        axis2.plot(xs, outputs[:, bi, 0], color="red")
        fig.savefig(os.path.join(task_dir, f"{bi}_left_wheel.svg"))
        plt.close(fig)

        fig = plt.figure()
        axis1 = fig.add_subplot(1, 1, 1)
        axis2 = axis1.twinx()
        axis1.plot(xs, features[:, bi, 1], color="green")
        axis1.plot(xs, features[:, bi, 4], color="blue")
        axis2.plot(xs, outputs[:, bi, 1], color="red")
        fig.savefig(os.path.join(task_dir, f"{bi}_right_wheel.svg"))
        plt.close(fig)

        fig = plt.figure()
        axis1 = fig.add_subplot(1, 1, 1)
        axis2 = axis1.twinx()
        axis1.plot(xs, features[:, bi, 2], color="green")
        axis1.plot(xs, features[:, bi, 5], color="blue")
        axis2.plot(xs, outputs[:, bi, 2], color="red")
        fig.savefig(os.path.join(task_dir, f"{bi}_pheromone_secretion.svg"))
        plt.close(fig)


def olfactory(task_dir):
    hist_path = os.path.join(task_dir, "history_77102066.npz")
    hist = Hist.load(hist_path)
    para = hist.get_min().min_para

    brain = RobotBrain(para)

    precision = 1000
    m = 10

    input_ = np.zeros(7)
    p = brain.get_mod_p()

    xs = np.arange(0, precision) / precision * m
    ys = np.zeros((precision, 3))

    for i in range(0, precision):
        xs[i] = input_[6] = m * i / precision
        ys[i, :] = p.calc(input_)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(xs, ys[:, 0], color="red")
    axis.plot(xs, ys[:, 1], color="blue")
    axis.plot(xs, ys[:, 2], color="green")
    fig.savefig(os.path.join(task_dir, "olfactory.svg"))
    plt.close(fig)


def dif_outputs(task_dir):
    features = np.load(os.path.join(task_dir, "features.npy"))
    outputs = np.load(os.path.join(task_dir, "outputs.npy"))

    features = (features[1:, :, :] - features[:-1, :, :]) / 0.033333
    outputs = (outputs[1:, :, :] - outputs[:-1, :, :]) / 0.033333

    xs = np.arange(0, features.shape[0]) * 0.033333

    for bi in range(9):
        fig = plt.figure()
        axis1 = fig.add_subplot(1, 1, 1)
        axis2 = axis1.twinx()
        axis1.plot(xs, features[:, bi, 0], color="green")
        axis1.plot(xs, features[:, bi, 3], color="blue")
        axis2.plot(xs, outputs[:, bi, 0], color="red")
        fig.savefig(os.path.join(task_dir, f"{bi}_dif_left_wheel.svg"))
        plt.close(fig)

        fig = plt.figure()
        axis1 = fig.add_subplot(1, 1, 1)
        axis2 = axis1.twinx()
        axis1.plot(xs, features[:, bi, 1], color="green")
        axis1.plot(xs, features[:, bi, 4], color="blue")
        axis2.plot(xs, outputs[:, bi, 1], color="red")
        fig.savefig(os.path.join(task_dir, f"{bi}_dif_right_wheel.svg"))
        plt.close(fig)

        fig = plt.figure()
        axis1 = fig.add_subplot(1, 1, 1)
        axis2 = axis1.twinx()
        axis1.plot(xs, features[:, bi, 2], color="green")
        axis1.plot(xs, features[:, bi, 5], color="blue")
        axis2.plot(xs, outputs[:, bi, 2], color="red")
        fig.savefig(os.path.join(task_dir, f"{bi}_dif_pheromone_secretion.svg"))
        plt.close(fig)


def symmetry(task_dir):
    features = np.load(os.path.join(task_dir, "features.npy"))
    outputs = np.load(os.path.join(task_dir, "outputs.npy"))

    skip = 1
    skip_index = int(skip / 0.033333 + 0.5)

    features = (features[skip_index + 1:, :, :] - features[skip_index:-1, :, :]) / 0.033333
    outputs = (outputs[skip_index + 1:, :, :] - outputs[skip_index:-1, :, :]) / 0.033333

    xs = np.arange(0, features.shape[0]) * 0.033333 + skip

    for bi in range(9):
        for k, n in enumerate(["left_wheel", "right_wheel", "pheromone"]):
            ave_p = features[:, bi, 0 + k] * outputs[:, bi, 0 + k]
            ave_p = np.sign(ave_p) * np.log(np.abs(ave_p) + 0.000001)
            ave_s = features[:, bi, 3 + k] * outputs[:, bi, 0 + k]
            ave_s = np.sign(ave_s) * np.log(np.abs(ave_s) + 0.000001)

            fig = plt.figure()
            axis = fig.add_subplot(2, 1, 1)
            axis.plot(xs, ave_p, color="green")
            axis = fig.add_subplot(2, 1, 2)
            axis.plot(xs, ave_s, color="blue")
            fig.savefig(os.path.join(task_dir, f"{bi}_sym_{n}.svg"))
            plt.close(fig)

    for k, n in enumerate(["left_wheel", "right_wheel", "pheromone"]):
        ave_p = np.average(features[:, :, 0:3] * outputs[:, :, :], axis=1)
        ave_p = np.sign(ave_p) * np.log(np.abs(ave_p) + 1)
        ave_s = np.average(features[:, :, 3:] * outputs[:, :, :], axis=1)
        ave_s = np.sign(ave_s) * np.log(np.abs(ave_s) + 1)

        fig = plt.figure()
        axis = fig.add_subplot(2, 1, 1)
        axis.plot(xs, ave_p[:, 0 + k], color="green")
        axis = fig.add_subplot(2, 1, 2)
        axis.plot(xs, ave_s[:, 0 + k], color="blue")

        fig.savefig(os.path.join(task_dir, f"sym_{n}.svg"))
        plt.close(fig)


def pheromone(task_dir):
    sensed_pheromone = np.load(os.path.join(task_dir, "sensed_pheromone.npy"))
    xs = np.arange(0, sensed_pheromone.shape[0]) * 0.033333

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    for bi in range(9):
        axis.plot(xs, sensed_pheromone[:, bi])
    fig.savefig(os.path.join(task_dir, "pheromone.svg"))
    plt.close(fig)


def plot_food_distance(task_dir):
    food_distance = np.load(os.path.join(task_dir, "food_distance.npy"))
    xs = np.arange(0, food_distance.shape[0]) * 0.033333

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    for i in range(2):
        axis.plot(xs, food_distance[:, i])
    fig.savefig(os.path.join(task_dir, "food_distance.svg"))
    plt.close(fig)


def main():
    cd = os.path.dirname(__file__)
    res_dir = os.path.join(cd, "results")

    for task_dir in os.listdir(res_dir):
        task_dir = os.path.join(res_dir, task_dir)

        if not os.path.isdir(task_dir):
            continue

        hist_path = os.path.join(task_dir, "history_77102066.npz")
        if not os.path.exists(hist_path):
            continue

        print(task_dir)

        # plot(task_dir)
        # dif_outputs(task_dir)
        # symmetry(task_dir)
        plot_food_distance(task_dir)
        # olfactory(task_dir)
        # pheromone(task_dir)


if __name__ == '__main__':
    main()
