import os

import matplotlib.pyplot as plt

from lib.utils import get_history
from lib.analizer.plot import plot_parameter_movements_graph


def main():
    project_directory = "../../../../"
    history = get_history(project_directory)
    fig = plot_parameter_movements_graph(history, 0, -1)
    fig.savefig(
        os.path.join(project_directory, "movement.svg")
    )
    plt.show()


if __name__ == '__main__':
    main()
