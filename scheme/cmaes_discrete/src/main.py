import os.path

import framework

from brain import BrainBuilder
from settings import Settings


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(save_dir, "result.pkl")

    settings = Settings()

    brain_builder = BrainBuilder(settings)
    framework.entry_points.train(settings, log_file_path, brain_builder)

    settings.Robot.ARGMAX_SELECTION = True
    framework.entry_points.record(settings, log_file_path, brain_builder)


if __name__ == '__main__':
    main()
