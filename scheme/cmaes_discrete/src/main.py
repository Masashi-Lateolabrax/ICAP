import os.path

import framework

from logger import Logger
from brain import BrainBuilder
from settings import Settings

import analysis


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_name = "result.pkl"

    settings = Settings()
    brain_builder = BrainBuilder(settings)
    logger = Logger(save_dir)

    ## Training.
    if not os.path.exists(os.path.join(save_dir, log_file_name)):
        framework.entry_points.train(settings, logger, brain_builder)
        logger.save(log_file_name)

    ## Record the training result in mp4 format.
    settings.Robot.ARGMAX_SELECTION = True
    if logger.is_empty():
        logger = Logger.load(os.path.join(save_dir, log_file_name))
    para = logger.get_min().min_ind
    dump = framework.entry_points.record(settings, save_dir, para, brain_builder, debug=True)

    analysis.plot_loss(settings, dump, os.path.join(save_dir, "loss.png"))


if __name__ == '__main__':
    main()
