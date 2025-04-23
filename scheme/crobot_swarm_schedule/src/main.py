import os.path
import datetime

from libs import utils
import framework

from logger import Logger
from brain import BrainBuilder
from settings import Settings

import analysis


def main():
    git_hash = utils.get_head_hash()[0:8]
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S-") + git_hash
    )
    # save_dir = "20250423_103010-7457c6a6"
    os.makedirs(save_dir, exist_ok=True)

    log_file_name1 = "result1.pkl"
    log_file_name2 = "result2.pkl"

    settings = Settings()
    brain_builder = BrainBuilder(settings)
    logger1 = Logger(save_dir)
    logger2 = Logger(save_dir)

    ## Training.
    if not os.path.exists(os.path.join(save_dir, log_file_name1)):
        tmp_generation = settings.CMAES.GENERATION

        settings.CMAES.GENERATION = int(tmp_generation * 0.2)
        framework.entry_points.train(settings, logger1, brain_builder)
        logger1.save(log_file_name1)

        settings.CMAES.GENERATION = int(tmp_generation * 0.8)
        settings.Robot.POSITION = []
        settings.Food.POSITION = []
        framework.entry_points.train(settings, logger2, brain_builder)
        logger2.save(log_file_name2)

        ## Load the logger if it is empty.
        if logger1.is_empty():
            logger1 = Logger.load(os.path.join(save_dir, log_file_name1))
        if logger2.is_empty():
            logger2 = Logger.load(os.path.join(save_dir, log_file_name2))

        analyze_results(logger1, os.path.join(save_dir, "stage1"), brain_builder)
        analyze_results(logger2, os.path.join(save_dir, "stage2"), brain_builder)


def analyze_results(logger, save_dir, brain_builder):
    ## Plot the movements of the parameters.
    file_path = os.path.join(save_dir, "parameter_movement.png")
    if not os.path.exists(file_path):
        analysis.plot_parameter_movements(file_path, logger)

    file_path = os.path.join(save_dir, "test_loss.png")
    if not os.path.exists(file_path):
        analysis.test_suboptimal_individuals(
            file_path, logger, brain_builder
        )

    file_path = os.path.join(save_dir, "videos")
    if not os.path.exists(file_path):
        analysis.record_in_mp4(
            file_path, logger, brain_builder
        )

    # analysis.plot_max_of_parameter(os.path.join(save_dir, "parameter_info.png"), logger, 0, 100)


if __name__ == '__main__':
    main()
