import matplotlib.pyplot as plt
import os
import time
import numpy as np

from experiment.evaporation_vs_liquid import Settings, AnalysisEnvironment, DataCollector


def collect(working_directory, liquid):
    collector = DataCollector(working_directory)
    task = AnalysisEnvironment(liquid)
    collector.run(task)
    return collector


def main(project_directory):
    num_trial = len(Settings.Task.LIQUID)
    max_evaporation = np.zeros(num_trial)

    for i, liquid in enumerate(Settings.Task.LIQUID):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        working_directory = os.path.join(project_directory, timestamp)
        if not os.path.exists(working_directory):
            os.mkdir(working_directory)
        data = collect(working_directory, liquid)

        x = np.arange(0, data.liquid.shape[0], 1) * Settings.Simulation.PHEROMONE_TIMESTEP
        dif_liquid = data.liquid[1:] - data.liquid[0:-1]
        dif_gas = data.gas[1:] - data.gas[0:-1]

        total_evaporation = np.sum(dif_liquid, axis=(1, 2))
        gas_stability = np.linalg.norm(dif_gas, axis=(1, 2))

        max_evaporation[i] = np.max(total_evaporation)

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.plot(x, total_evaporation)
        fig.savefig(os.path.join(working_directory, "total_evaporation.svg"))

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.plot(x, gas_stability)
        fig.savefig(os.path.join(working_directory, "gas_stability.svg"))

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot([i for i in range(num_trial)], max_evaporation)
    fig.savefig(os.path.join(project_directory, "evaporation_vs_liquid.svg"))


if __name__ == '__main__':
    main(os.path.dirname(__file__))
