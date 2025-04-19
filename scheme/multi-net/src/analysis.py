import framework
from framework.anaylysis import record_in_mp4, plot_parameter_movements

from loss import Loss
from settings import Settings
from brain import BrainBuilder
from logger import Logger


def plot_loss(file_path: str, dump: framework.Dump):
    framework.anaylysis.plot_loss(
        file_path,
        Settings(),
        dump,
        Loss()
    )


def test_suboptimal_individuals(
        save_dir: str,
        logger: Logger,
        brain_builder: BrainBuilder,
):
    settings = Settings()
    settings.Robot.ARGMAX_SELECTION = True
    task_generator = framework.TaskGenerator(settings, brain_builder)
    return framework.anaylysis.test_suboptimal_individuals(
        save_dir,
        Settings(),
        logger,
        task_generator
    )
