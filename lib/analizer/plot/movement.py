import numpy as np
import matplotlib.pyplot as plt

from src.lib import Hist
from lib.utils import get_current_history


def main():
    history = Hist.load(
        get_current_history("../../../")
    )
    dim = range(0, history.dim)  # range(0, history.dim)
    generation = range(0, len(history.queues))  # range(0, len(history.queues))

    n = generation.stop - generation.start

    movements = np.zeros((n - 1,))
    for i in range(0, n - 1):
        s = history.queues[generation.start + i + 1].min_para - history.queues[generation.start + i].min_para
        movements[i] = np.linalg.norm(s)

    parameters = history.queues[generation.stop - 1].min_para
    std = np.std(parameters)

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    x = np.arange(generation.start, generation.stop - 1, dtype=np.uint64)
    ax1.plot(x, movements)
    ax1.set_xlabel("generations")
    ax1.set_ylabel("movements")

    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(dim.start, dim.stop, dtype=np.uint64)
    ax2.bar(x, history.queues[generation.stop - 1].min_para)
    ax2.plot(x, [std * 3] * len(x), c="Red")
    ax2.plot(x, [std * -3] * len(x), c="Red")
    ax2.set_xlabel("parameters")
    ax2.set_ylabel("values")

    plt.show()


if __name__ == '__main__':
    main()
