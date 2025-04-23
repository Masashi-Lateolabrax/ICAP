from libs.optimizer import Individual

import framework
from framework.analysis import plot_parameter_movements, LabelAndColor

from loss import Loss
from settings import Settings
from brain import BrainBuilder
from logger import Logger


def plot_loss(file_path: str, ind: Individual):
    framework.analysis.plot_loss(
        file_path,
        Settings(),
        ind,
        Loss(),
        labels={
            0: LabelAndColor(label="robot and food", color="blue"),
            1: LabelAndColor(label="food and nest", color="orange"),
        }
    )


def record_in_mp4(save_dir: str, logger: Logger, brain_builder: BrainBuilder):
    settings = Settings()
    settings.Robot.ARGMAX_SELECTION = True

    framework.analysis.record_in_mp4(
        save_dir,
        settings,
        logger,
        Loss(),
        brain_builder,
        labels={
            0: LabelAndColor(label="robot and food", color="blue"),
            1: LabelAndColor(label="food and nest", color="orange"),
        }
    )


def test_suboptimal_individuals(
        file_path: str,
        logger: Logger,
        brain_builder: BrainBuilder,
):
    settings = Settings()
    settings.Robot.ARGMAX_SELECTION = True

    task_generator = framework.TaskGenerator(settings, brain_builder)

    return framework.analysis.test_suboptimal_individuals(
        file_path,
        Settings(),
        logger,
        task_generator
    )


def plot_max_of_parameter(
        file_path: str,
        logger: Logger,
        start: int = 0,
        end: int = None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    end = end if end is not None else len(logger)
    individuals = [logger[i].min_ind for i in range(start, end)]

    max_values = [np.max(ind) for ind in individuals]
    min_values = [np.min(ind) for ind in individuals]
    ave_values = [np.mean(ind) for ind in individuals]
    std_values = [np.std(ind) for ind in individuals]
    generation = np.arange(len(max_values))

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(generation, max_values, label="max of parameter")
    axis.plot(generation, min_values, label="min of parameter")
    axis.plot(generation, ave_values, label="ave of parameter")
    axis.fill_between(
        generation,
        np.array(ave_values) - np.array(std_values),
        np.array(ave_values) + np.array(std_values),
        alpha=0.2,
        label="std of parameter"
    )
    plt.savefig(file_path)
