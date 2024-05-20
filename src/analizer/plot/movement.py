import numpy as np
import matplotlib.pyplot as plt

from src.optimizer import Hist
from src.utils import get_current_history


def main():
    history = Hist.load(
        get_current_history("../../")
    )

    prev_point = history.queues[0].min_para
    x = np.arange(1, len(history.queues), dtype=np.uint)
    data = np.zeros((len(history.queues) - 1,))
    for i, q in enumerate(history.queues):
        if i == 0:
            continue
        data[i - 1] = np.linalg.norm(prev_point - q.min_para)
        prev_point = q.min_para

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, data, c="Blue")
    plt.show()


if __name__ == '__main__':
    main()
