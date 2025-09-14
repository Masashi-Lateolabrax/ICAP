import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from framework.prelude import *


def plot_pheromone_history(
        filepath: str,
        recorder: IndividualRecorder,
):
    if os.path.exists(filepath):
        print(f"File {filepath} already exists.")
        return

    pheromone = []
    for rec in recorder:
        best_individual: Individual = rec.best_individual
        pheromone.append(
            best_individual.get_max_gas_pheromone()
        )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(np.arange(len(pheromone)), pheromone, label="Max Gas Pheromone", color='green')

    ax.set_xlabel("Generation")
    ax.set_ylabel("Max Gas Pheromone")
    ax.set_title("Max Gas Pheromone Over Generations")

    fig.savefig(filepath)
