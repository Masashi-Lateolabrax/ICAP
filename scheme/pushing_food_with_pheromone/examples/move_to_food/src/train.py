import os.path

from libs.optimizer import CMAES, Hist, OneThreadProc

from .prerulde import Settings

from .brain import Brain
from .task import TaskGenerator


def train(log_path):
    env_creator = TaskGenerator()

    logger = Hist(os.path.dirname(log_path))
    cmaes = CMAES(
        dim=Brain.get_dim(),
        generation=Settings.CMAES_GENERATION,
        population=Settings.CMAES_POPULATION,
        mu=Settings.CMAES_MU,
        logger=logger
    )
    cmaes.optimize(env_creator)

    logger.save(os.path.basename(log_path))
