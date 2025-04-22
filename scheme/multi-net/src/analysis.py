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
