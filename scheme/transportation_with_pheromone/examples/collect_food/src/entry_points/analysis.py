import numpy as np
import matplotlib.pyplot as plt

from ..logger import Logger


def analyze(log_path):
    logger = Logger.load(log_path)

    generation = -1

    q = logger._logger.queues[generation]
    the_most_min_idx = np.argmin(q.loss)
    loss_fr = np.cumsum(q.loss_fr[the_most_min_idx, :])
    loss_fn = np.cumsum(q.loss_fn[the_most_min_idx, :])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(loss_fr, color='r')
    ax2.plot(loss_fn, color='b')

    ax1.set_ylabel("Food and Robot [RED]")
    ax2.set_ylabel("Food and Nest [BLUE]")

    fig.savefig(f"gen{generation}.pdf")
