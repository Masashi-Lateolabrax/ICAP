from lib.optimizer import Hist


def plot_parameter_movements_3d_graph(history: Hist, start: int, stop: int):
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np

    dim = range(0, history.dim)
    generation = range(start, stop if start < stop < len(history.queues) else len(history.queues))

    num_gen = generation.stop - generation.start
    num_dim = dim.stop - dim.start

    subs = np.zeros((num_dim, num_gen - 1))
    for i in range(0, num_gen - 1):
        s = history.queues[generation.start + i + 1].min_para - history.queues[generation.start + i].min_para
        subs[:, i] = np.abs(s[dim])
        print(f"{i}/{num_gen - 2}")

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

    # Tile version
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

    plt.show()


def main():
    import os

    working_directory = "../../../src/"

    # history = get_current_history(working_directory)
    history = Hist.load(
        os.path.join(working_directory, "history_8a1d803b.npz")
    )

    plot_parameter_movements_3d_graph(history, 0, -1)


if __name__ == '__main__':
    main()
