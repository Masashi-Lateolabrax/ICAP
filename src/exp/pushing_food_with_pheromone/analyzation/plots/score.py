import os

import matplotlib.pyplot as plt

from lib.analizer.plot import plot_score_graph
from lib.utils import get_history


def main():
    project_directory = "../../../../"
    history = get_history(project_directory)
    fig = plot_score_graph(history)
    fig.savefig(
        os.path.join(project_directory, "movement.svg")
    )
    plt.show()


if __name__ == '__main__':
    main()
