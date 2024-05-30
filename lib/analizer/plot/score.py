import numpy as np
import matplotlib.pyplot as plt

from src.lib import Hist
from lib.utils import get_current_history


def main():
    history = Hist.load(
        get_current_history("../../../")
    )

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


if __name__ == '__main__':
    main()
