import os.path
import datetime

from libs import utils
import framework

from logger import Logger
from brain import BrainBuilder
from settings import Settings

import analysis


def main():
    git_hash = utils.get_head_hash()
    # save_dir = os.path.join(
    #     os.path.dirname(os.path.abspath(__file__)),
    #     datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + git_hash
    # )
    save_dir = "20250408_050158_a3ae57fa"
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
    file_path = os.path.join(save_dir, "parameter_movements.png")
    if not os.path.exists(file_path):
        analysis.plot_parameter_movements(logger, file_path)

    ## Loop through the generations.
    settings.Robot.ARGMAX_SELECTION = True
    for gen in [logger.get_min_gen()]:
        para = logger[gen].min_ind
        if gen == logger.get_min_gen():
            file_path = os.path.join(save_dir, f"gen{gen}(best).png")
        else:
            file_path = os.path.join(save_dir, f"gen{gen}.png")

        ## Record the simulation.
        dump = framework.entry_points.record(
            settings,
            file_path,
            para,
            brain_builder,
            debug=False
        )

        ## Plot the loss.
        analysis.plot_loss(settings, dump, os.path.join(save_dir, f"loss_gen{gen}.png"))


if __name__ == '__main__':
    main()
