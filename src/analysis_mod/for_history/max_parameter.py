import matplotlib.pyplot as plt
import numpy as np

from framework.prelude import *


def plot_max_value_in_parameters(saved_individuals: IndividualRecorder, filepath: str, top_n=1):
    generations = np.arange(len(saved_individuals))
    values = np.zeros((top_n, len(saved_individuals)), dtype=float)
    index = np.zeros((top_n, len(saved_individuals)), dtype=int)

    for i, rec in enumerate(saved_individuals):
        rec: Rec = rec
        abs_best_individual = [(i, abs(v), v) for i, v in enumerate(rec.best_individual)]
        sored_parameters = sorted(abs_best_individual, key=lambda x: x[1], reverse=True)

        for j in range(min(top_n, len(sored_parameters))):
            values[j, i] = sored_parameters[j][2]
            index[j, i] = sored_parameters[j][0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    for i in range(top_n):
        if i < values.shape[0]:
            ax1.plot(generations, values[i], label=f'Value {i}')
            ax2.scatter(generations, index[i], label=f'Index {i}')

    plt.savefig(filepath)


def parameter_heatmap(saved_individuals: IndividualRecorder, filepath: str):
    generations = np.arange(len(saved_individuals))
    num_parameters = len(saved_individuals[0].best_individual)

    heatmap_data = np.zeros((num_parameters, len(saved_individuals)), dtype=float)

    for i, rec in enumerate(saved_individuals):
        rec: Rec = rec
        best_ind = rec.best_individual
        max_value = np.max(best_ind)
        for j in range(num_parameters):
            heatmap_data[j, i] = best_ind[j] / max_value

    fig, ax = plt.subplots()
    cax = ax.imshow(heatmap_data, aspect='auto', cmap='hot', interpolation='nearest')

    ax.set_title('Parameter Heatmap Over Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Parameter Index')
    ax.set_xticks(np.arange(len(generations)))
    ax.set_yticks(np.arange(num_parameters))
    ax.set_xticklabels(generations)
    ax.set_yticklabels(np.arange(num_parameters))

    fig.colorbar(cax)
    plt.savefig(filepath)
