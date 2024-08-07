import warnings
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

if __name__ == "__main__":
    from experiment.SV_vs_EV_speed_curve import Settings, AnalysisEnvironment, DataCollector
else:
    from .experiment.SV_vs_EV_speed_curve import Settings, AnalysisEnvironment, DataCollector


def collect_data(parent_working_directory):
    properties = np.zeros((len(Settings.Task.SATURATION_VAPOR), 3))
    for i, sv in enumerate(Settings.Task.SATURATION_VAPOR):
        working_directory = os.path.join(parent_working_directory, f"sv_{sv:03.3f}")
        if os.path.exists(working_directory):
            warnings.warn(f"The task where SV is {sv}, has already been calculated.")
            continue
        os.mkdir(working_directory)

        evaporation_speeds = np.zeros((2, len(Settings.Task.EVAPORATION)))
        for j, e in enumerate(Settings.Task.EVAPORATION):
            task_directory = os.path.join(working_directory, f"ev_{e:03.3f}")
            if os.path.exists(task_directory):
                warnings.warn(f"The task where SV and EV are {sv} and {e} respectively, has already been calculated.")
                continue
            os.mkdir(task_directory)

            collector = DataCollector(task_directory)
            task = AnalysisEnvironment(e, sv)
            collector.run(task)
            collector.release()
            np.save(os.path.join(task_directory, "def_liquid"), collector.dif_liquid)
            np.save(os.path.join(task_directory, "gas"), collector.gas)

            dif_gas = collector.gas[1:, :, :] - collector.gas[:-1, :, :]
            gas_stability = np.exp(-np.linalg.norm(dif_gas, axis=(1, 2)))
            stable_index = np.argmax(gas_stability)

            evaporation_speeds[0, j] = e
            evaporation_speeds[1, j] = collector.dif_liquid[stable_index]

        def func(x, a, b, c):
            return a * np.log(x) + b * x + c

        optimized_parameter, _ = curve_fit(func, evaporation_speeds[0], evaporation_speeds[1])
        properties[i] = optimized_parameter

        popt, _ = curve_fit(func, evaporation_speeds[0], evaporation_speeds[1])

        new_x = np.linspace(evaporation_speeds[0, 0], evaporation_speeds[0, -1], evaporation_speeds.shape[1] * 10)

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(f"{popt[0]:6g}Ln(x)+{popt[1]:6g}x+{popt[2]:6g}")
        axis.scatter(evaporation_speeds[0], evaporation_speeds[1], label='Data')
        axis.plot(new_x, func(new_x, *popt), label='func')
        fig.savefig(os.path.join(working_directory, "speed_vs_coefficient(curve_fit).svg"))

    np.save(os.path.join(parent_working_directory, "func_properties"), properties)


def create_and_save_graph(working_directory):
    optimized_parameter = np.load(os.path.join(working_directory, "func_properties.npy"))
    xs = np.array(Settings.Task.SATURATION_VAPOR)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(xs, optimized_parameter[:, 0])
    axis.plot(xs, optimized_parameter[:, 1])
    axis.plot(xs, optimized_parameter[:, 2])
    fig.savefig(os.path.join(working_directory, "sv_vs_speed_curve_property.svg"))

    def func(x, a, b):
        return a * x + b

    for p in [optimized_parameter[:, i] for i in range(optimized_parameter.shape[1])]:
        print(curve_fit(func, xs, p)[0])


def main(timestamp: str, project_directory: str):
    parent_working_directory: str = os.path.join(project_directory, timestamp)

    if not os.path.exists(parent_working_directory):
        os.mkdir(parent_working_directory)
        collect_data(parent_working_directory)
    else:
        warnings.warn("This task has been already calculated.")

    create_and_save_graph(parent_working_directory)


if __name__ == '__main__':
    import time

    main(time.strftime("%Y%m%d_%H%M%S"), os.path.dirname(__file__))
