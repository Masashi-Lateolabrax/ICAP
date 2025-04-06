import os.path

import framework

from logger import Logger
from brain import BrainBuilder
from settings import Settings

import analysis


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_name = "result.pkl"

    logger = Logger(save_dir)
    settings = Settings()

    brain_builder = BrainBuilder(settings)
    framework.entry_points.train(settings, logger, brain_builder)
    logger.save(log_file_name)

    settings.Robot.ARGMAX_SELECTION = True
    para = logger.get_min().min_para
    dump = framework.entry_points.record(settings, save_dir, para, brain_builder)

    analysis.plot_loss(settings, dump, save_dir)


if __name__ == '__main__':
    main()
