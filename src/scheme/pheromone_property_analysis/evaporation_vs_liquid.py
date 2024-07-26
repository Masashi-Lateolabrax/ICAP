import matplotlib.pyplot as plt
import os
import time
import numpy as np

if __name__ == "__main__":
    from experiment.evaporation_vs_liquid import Settings, AnalysisEnvironment, DataCollector
else:
    from .experiment.evaporation_vs_liquid import Settings, AnalysisEnvironment, DataCollector


def collect(working_directory, liquid):
    collector = DataCollector(working_directory)
    task = AnalysisEnvironment(liquid)
    collector.run(task)
    collector.release()
    return collector


def create_and_save_graph(working_directory, dif_liquid, gas):
    center_index = np.array(Settings.World.CENTER_INDEX)

    dif_gas = gas[1:, :, :] - gas[:-1, :, :]
    gas_stability = np.exp(-np.linalg.norm(dif_gas, axis=(1, 2)))
    max_gas = np.max(gas, axis=(1, 2))
    pheromone_size = np.zeros(gas.shape[0])

    secretion_stop_index = np.max(
        np.where(gas_stability[:int(Settings.Task.SECRETION_PERIOD / Settings.Simulation.TIMESTEP)] > 0.999)
    )

    effective_max_gas = max_gas[:gas_stability.shape[0]] * np.where(gas_stability > 0.999, 1.0, np.NAN)
    emg_max = np.average(
        effective_max_gas[:secretion_stop_index][np.logical_not(np.isnan(effective_max_gas[:secretion_stop_index]))]
    )
    emg_min = np.average(
        effective_max_gas[secretion_stop_index:][np.logical_not(np.isnan(effective_max_gas[secretion_stop_index:]))]
    )

    decreasing = max_gas[secretion_stop_index:][max_gas[secretion_stop_index:] > emg_max * 0.5]

    for t in range(gas.shape[0]):
        uppers = np.where(gas[t, :, :] >= max_gas[t] * 0.5)
        indexes = np.array(uppers).T
        distances = np.linalg.norm(indexes - center_index, axis=1)
        pheromone_size[t] = np.max(distances)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(
        [i * Settings.Simulation.TIMESTEP for i in range(dif_liquid.shape[0])],
        dif_liquid
    )
    axis.set_title(f"speed={dif_liquid[secretion_stop_index]}")
    fig.savefig(os.path.join(working_directory, "dif_liquid.svg"))

    fig = plt.figure()
    axis1 = fig.add_subplot(1, 1, 1)
    axis1.set_ylabel("stability")
    axis1.plot(
        [i * Settings.Simulation.TIMESTEP for i in range(gas_stability.shape[0])],
        gas_stability
    )
    fig.savefig(os.path.join(working_directory, "gas_stability.svg"))

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(
        [i * Settings.Simulation.TIMESTEP for i in range(max_gas.shape[0])],
        max_gas
    )
    fig.savefig(os.path.join(working_directory, "max_gas.svg"))

    fig = plt.figure()
    axis1 = fig.add_subplot(1, 1, 1)
    axis1.set_title(f"max={np.average(emg_max)}, min={np.average(emg_min)}")
    axis1.set_ylabel("effective max gas")
    axis1.plot(
        [i * Settings.Simulation.TIMESTEP for i in range(effective_max_gas.shape[0])],
        effective_max_gas,
    )
    fig.savefig(os.path.join(working_directory, "effective_max_gas.svg"))

    fig = plt.figure()
    xs = [(i + secretion_stop_index) * Settings.Simulation.TIMESTEP for i in range(0, decreasing.shape[0])]
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(xs, [emg_max for _ in range(len(xs))])
    axis.plot(xs, decreasing)
    axis.set_title(f"Half-life = {len(xs) * Settings.Simulation.TIMESTEP}")
    fig.savefig(os.path.join(working_directory, "max_gas_decreasing.svg"))

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(
        [i * Settings.Simulation.TIMESTEP for i in range(pheromone_size.shape[0])],
        pheromone_size
    )
    fig.savefig(os.path.join(working_directory, "pheromone_size.svg"))

    plt.close('all')


def main(project_directory):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # timestamp = "20240726_183852"

    working_directory = os.path.join(project_directory, timestamp)
    if not os.path.exists(working_directory):
        os.mkdir(working_directory)

        for i, liquid in enumerate(Settings.Task.LIQUID):
            task_directory = os.path.join(working_directory, str(i))
            os.mkdir(task_directory)
            data = collect(task_directory, liquid)
            np.save(os.path.join(task_directory, "dif_liquid"), data.dif_liquid)
            np.save(os.path.join(task_directory, "gas"), data.gas)

    for i in range(len(Settings.Task.LIQUID)):
        task_directory = os.path.join(working_directory, str(i))
        dif_liquid = np.load(os.path.join(task_directory, "dif_liquid.npy"))
        gas = np.load(os.path.join(task_directory, "gas.npy"))
        create_and_save_graph(task_directory, dif_liquid, gas)


if __name__ == '__main__':
    main(os.path.dirname(__file__))
