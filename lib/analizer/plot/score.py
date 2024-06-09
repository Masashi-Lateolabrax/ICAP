from lib.optimizer import Hist


def plot_score_graph(history: Hist):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(0, len(history.queues), dtype=np.uint)
    data = np.zeros((len(history.queues), 2))
    for i, q in enumerate(history.queues):
        data[i, 0] = q.min_score
        data[i, 1] = q.max_score

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, data[:, 0], c="Blue")
    ax.plot(x, data[:, 1], c="Red")
    plt.show()


def main():
    from lib.utils import get_history

    working_directory = "../../../src/"

    history = get_history(working_directory)

    plot_score_graph(history)


if __name__ == '__main__':
    main()
