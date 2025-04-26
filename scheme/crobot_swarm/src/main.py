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

    log_file_name = "result.pkl"

    settings = Settings()
    brain_builder = BrainBuilder(settings)
    logger = Logger(save_dir)

    ## Training.
    if not os.path.exists(os.path.join(save_dir, log_file_name)):
        framework.entry_points.train(settings, logger, brain_builder)
        logger.save(log_file_name)

    ## Load the logger if it is empty.
    if logger.is_empty():
        logger = Logger.load(os.path.join(save_dir, log_file_name))

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
