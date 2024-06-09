from lib.optimizer import Hist


def plot_parameter_movements_graph(history: Hist, start: int, end: int):
    import matplotlib.pyplot as plt
    import numpy as np

    end = end if start < end <= len(history.queues) else len(history.queues)
    queues = history.queues[start:end]
    n = len(queues) - 1
    movements = np.zeros((n,))
    x = np.arange(start=start, stop=end - 1, dtype=np.uint64)

    for i in range(0, n):
        sub = queues[i + 1].min_para - queues[i].min_para
        movements[i] = np.linalg.norm(sub)

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, movements)
    ax.set_xlabel("generations")
    ax.set_ylabel("movements")

    plt.show()


def main():
    import os

    working_directory = "../../../src/"

    # history = get_current_history(working_directory)
    history = Hist.load(
        os.path.join(working_directory, "history_8a1d803b.npz")
    )

    plot_parameter_movements_graph(history, 0, -1)


if __name__ == '__main__':
    main()
