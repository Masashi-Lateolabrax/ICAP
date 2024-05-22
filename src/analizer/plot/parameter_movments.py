import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from src.optimizer import Hist
from src.utils import get_current_history


def main():
    history = Hist.load(
        get_current_history("../../")
    )
    dim = range(0, history.dim)  # range(0, history.dim)
    generation = range(0, len(history.queues))  # range(0, len(history.queues))

    num_gen = generation.stop - generation.start
    num_dim = dim.stop - dim.start

    subs = np.zeros((num_dim, num_gen - 1))
    for i in range(0, num_gen - 1):
        s = history.queues[generation.start + i + 1].min_para - history.queues[generation.start + i].min_para
        subs[:, i] = np.abs(s[dim])
        print(f"{i}/{num_gen - 2}")

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # cmap = plt.get_cmap('tab20').colors
    # colors = [cmap[i % len(cmap)] for i in range(0, history.dim)]
    # x = np.arange(generation.start, generation.start + subs.shape[1], dtype=np.uint64)
    # for i, (d, c) in enumerate(zip(reversed(dim), colors)):
    #     ax1.bar(x, subs[d], zs=d, zdir="y", alpha=1, color=c)
    #     print(f"{i}/{num_dim - 1}")
    # ax1.set_xlabel("generation")
    # ax1.set_ylabel("parameters")
    # ax1.set_zlabel("movement")

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    x, y = np.meshgrid(
        np.arange(generation.start, generation.stop - 1),
        np.arange(dim.start, dim.stop)
    )
    ax1.plot_surface(
        x, y, subs,
        rstride=1, cstride=1, linewidth=0,
        cmap=cm.coolwarm, antialiased=False
    )
    ax1.set_xlabel("generation")
    ax1.set_ylabel("parameters")
    ax1.set_zlabel("movement")

    plt.show()


if __name__ == '__main__':
    main()
