import matplotlib.pyplot as plt
import os
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
    stable_index = np.argmax(gas_stability)
    max_gas = np.max(gas, axis=(1, 2))
    pheromone_size = np.zeros(gas.shape[0])

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
    axis.set_title(f"stability={gas_stability[stable_index]}, speed={dif_liquid[stable_index]}")
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
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(
        [i * Settings.Simulation.TIMESTEP for i in range(pheromone_size.shape[0])],
        pheromone_size
    )
    fig.savefig(os.path.join(working_directory, "pheromone_size.svg"))

    plt.close('all')

    return dif_liquid[stable_index]


def main(timestamp: str, project_directory: str):
    working_directory: str = os.path.join(project_directory, timestamp)

    if not os.path.exists(working_directory):
        os.mkdir(working_directory)

        for i, evaporation in enumerate(Settings.Task.EVAPORATION):
            task_directory = os.path.join(working_directory, str(i))
            os.mkdir(task_directory)
            data = collect(task_directory, evaporation)
            np.save(os.path.join(task_directory, "dif_liquid"), data.dif_liquid)
            np.save(os.path.join(task_directory, "gas"), data.gas)

    evaporation_speeds = []

    for i in range(len(Settings.Task.EVAPORATION)):
        task_directory = os.path.join(working_directory, str(i))
        dif_liquid = np.load(os.path.join(task_directory, "dif_liquid.npy"))
        gas = np.load(os.path.join(task_directory, "gas.npy"))
        evaporation_speed = create_and_save_graph(task_directory, dif_liquid, gas)
        evaporation_speeds.append(evaporation_speed)

    fig = plt.figure()
    data = np.array([Settings.Task.EVAPORATION, evaporation_speeds])
    np.save(os.path.join(working_directory, "speed_vs_coefficient.npy"), data)
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(data[0], data[1])
    fig.savefig(os.path.join(working_directory, "speed_vs_coefficient.svg"))


def curve_fitting(timestamp: str, project_directory: str):
    working_directory = os.path.join(project_directory, timestamp)

    def func(x, a, b, c):
        return a * np.log(x) + b * x + c

    from scipy.optimize import curve_fit

    data: np.ndarray = np.load(os.path.join(working_directory, "speed_vs_coefficient.npy"))
    popt, _ = curve_fit(func, data[0], data[1])
    xs = np.arange(300, 20000, dtype=np.float64) / 100

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(f"{popt[0]:6g}Ln(x)+{popt[1]:6g}x+{popt[2]:6g}")
    axis.scatter(data[0], data[1], label='Data')
    axis.plot(xs, func(xs, *popt), label='func')
    fig.savefig(os.path.join(working_directory, "speed_vs_coefficient(curve_fit).svg"))


if __name__ == '__main__':
    import time
    main(time.strftime("%Y%m%d_%H%M%S"), os.path.dirname(__file__))
    # main("20240727_095058", os.path.dirname(__file__))
    # curve_fitting("20240729_003124", os.path.dirname(__file__))
