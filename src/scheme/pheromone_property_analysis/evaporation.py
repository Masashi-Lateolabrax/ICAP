import matplotlib.pyplot as plt
import os
import time
import numpy as np

if __name__ == "__main__":
    from experiment.evaporation import Settings, AnalysisEnvironment, DataCollector
else:
    from .experiment.evaporation import Settings, AnalysisEnvironment, DataCollector


def collect(working_directory, evaporation):
    collector = DataCollector(working_directory)
    task = AnalysisEnvironment(evaporation)
    collector.run(task)
    collector.release()
    return collector


def create_and_save_graph(working_directory, dif_liquid, gas):
    center_index = np.array(Settings.World.CENTER_INDEX)

    dif_gas = gas[1:, :, :] - gas[:-1, :, :]
    gas_stability = np.exp(-np.linalg.norm(dif_gas, axis=(1, 2)))
    max_gas = np.max(gas, axis=(1, 2))
    pheromone_size = np.zeros(gas.shape[0])

    secretion_stop_index = np.max(np.where(gas_stability > 0.999))

    effective_max_gas = max_gas[:gas_stability.shape[0]] * np.where(gas_stability > 0.999, 1.0, np.NAN)
    emg_max = np.average(
        effective_max_gas[:secretion_stop_index][np.logical_not(np.isnan(effective_max_gas[:secretion_stop_index]))]
    )
    emg_min = np.average(
        effective_max_gas[secretion_stop_index:][np.logical_not(np.isnan(effective_max_gas[secretion_stop_index:]))]
    )

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
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(
        [i * Settings.Simulation.TIMESTEP for i in range(pheromone_size.shape[0])],
        pheromone_size
    )
    fig.savefig(os.path.join(working_directory, "pheromone_size.svg"))

    plt.close('all')

    return emg_max, emg_min


def main(project_directory):
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp = "20240727_095058"

    working_directory = os.path.join(project_directory, timestamp)
    if not os.path.exists(working_directory):
        os.mkdir(working_directory)

        for i, evaporation in enumerate(Settings.Task.EVAPORATION):
            task_directory = os.path.join(working_directory, str(i))
            os.mkdir(task_directory)
            data = collect(task_directory, evaporation)
            np.save(os.path.join(task_directory, "dif_liquid"), data.dif_liquid)
            np.save(os.path.join(task_directory, "gas"), data.gas)

    emg_max_list = []
    emg_min_list = []

    for i in range(len(Settings.Task.EVAPORATION)):
        task_directory = os.path.join(working_directory, str(i))
        dif_liquid = np.load(os.path.join(task_directory, "dif_liquid.npy"))
        gas = np.load(os.path.join(task_directory, "gas.npy"))
        emg_max, emg_min = create_and_save_graph(task_directory, dif_liquid, gas)
        emg_max_list.append(emg_max)
        emg_min_list.append(emg_min)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(Settings.Task.EVAPORATION, emg_max_list)
    axis.plot(Settings.Task.EVAPORATION, emg_min_list)
    fig.savefig(os.path.join(working_directory, "speed_vs_coefficient.svg"))


if __name__ == '__main__':
    main(os.path.dirname(__file__))
