import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from .settings import Settings, EType
from .logger import LogLoader


def create_and_save_fig(
        path: str,
        plotter: Callable[[plt.Figure], None],
        width=8,  # the unit for this parameter is centimeters
        aspect_rate=4 / 3,
        font_size=10,
        dpi=300,
):
    with plt.rc_context({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{sfmath}",
        "font.family": "BIZ UDPGothic",
        "font.size": font_size,
        "figure.dpi": dpi
    }):
        cm_per_inch = 2.54
        fig = plt.figure(figsize=(width / cm_per_inch, (width / cm_per_inch) / aspect_rate))
        plotter(fig)
        fig.savefig(path, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)


def plot_evaluation(work_dir, evaluation):
    def plotter(fig: plt.Figure):
        nonlocal evaluation

        xs = np.arange(0, Settings.Simulation.TOTAL_STEP) * Settings.Simulation.TIMESTEP

        axis = fig.add_subplot(1, 1, 1)
        axis.plot(xs, evaluation)

        axis.set_title("Evaluation")

    create_and_save_fig(
        os.path.join(work_dir, "evaluation.pdf"),
        plotter,
    )


def plot_pheromone_gas_volume(work_dir, pheromone_gas):
    def plotter(fig: plt.Figure):
        nonlocal pheromone_gas

        total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)
        xs = np.arange(0, total_step) * Settings.Simulation.TIMESTEP

        axis = fig.add_subplot(1, 1, 1)

        axis.plot(xs, pheromone_gas)
        axis.set_title("Pheromone Gas Volume")

    create_and_save_fig(
        os.path.join(work_dir, "pheromone_gas_volume.pdf"),
        plotter,
    )


def plot_evaluation_elements_for_each_generation(workdir, loader: LogLoader):
    evaluations = np.zeros((Settings.Optimization.GENERATION, 3))

    for gen in range(len(loader)):
        inds = loader.get_individuals(gen)

        dump = np.array([i.dump for i in inds])
        dump = np.sum(dump, axis=1)
        summed_dump = np.sum(dump, axis=2)

        if Settings.Optimization.EVALUATION_TYPE == EType.POTENTIAL:
            i = np.argmax(summed_dump[:, EType.POTENTIAL])
            evaluations[gen, 0:2] = dump[i, EType.POTENTIAL, :]
            evaluations[gen, 2] = np.sum(evaluations[gen, 0:2])

        elif Settings.Optimization.EVALUATION_TYPE == EType.DISTANCE:
            i = np.argmin(summed_dump[:, EType.DISTANCE])
            evaluations[gen, 0:2] = dump[i, EType.DISTANCE, :]
            evaluations[gen, 2] = np.sum(evaluations[gen, 0:2])

        else:
            raise Exception("Selected an invalid EVALUATION_TYPE.")

    def plotter(fig: plt.Figure):
        nonlocal evaluations

        evaluations /= Settings.Simulation.TOTAL_STEP
        xs = np.arange(0, evaluations.shape[0])

        axis = fig.add_subplot(1, 1, 1)
        axis.plot(xs, evaluations[:, 2], c="#d3d3d3", label="SUM")
        axis.plot(xs, evaluations[:, 0], c="#0000ff", label="Food-Robot")
        axis.plot(xs, evaluations[:, 1], c="#ff0000", label="Nest-Food")

        axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    create_and_save_fig(
        os.path.join(workdir, "evaluation_elements_for_each_generation.pdf"),
        plotter,
    )


def evaluation_for_each_generation(workdir, loader: LogLoader):
    e_type = Settings.Optimization.EVALUATION_TYPE
    evaluations = np.zeros((Settings.Optimization.GENERATION, 3))

    for gen in range(len(loader)):
        inds = loader.get_individuals(gen)

        dump = np.array([i.dump for i in inds])
        dump = np.sum(dump, axis=1)
        summed_dump = np.sum(dump, axis=2)

        min_i = np.argmin(summed_dump[:, e_type])
        min_score = summed_dump[min_i, e_type]

        max_i = np.argmax(summed_dump[:, e_type])
        max_score = summed_dump[max_i, e_type]

        ave_score = np.average(summed_dump[:, e_type])

        evaluations[gen, 0] = min_score
        evaluations[gen, 1] = max_score
        evaluations[gen, 2] = ave_score

    def plotter(fig: plt.Figure):
        nonlocal evaluations

        evaluations /= Settings.Simulation.TOTAL_STEP
        xs = np.arange(0, evaluations.shape[0])

        axis = fig.add_subplot(1, 1, 1)
        axis.plot(xs, evaluations[:, 2])
        axis.fill_between(xs, evaluations[:, 0], evaluations[:, 1], color="gray", alpha=0.3)

        axis.set_ylabel("Evaluation")
        axis.set_xlabel("Generation")

    create_and_save_fig(
        os.path.join(workdir, "evaluation_for_each_generation.pdf"),
        plotter,
    )
