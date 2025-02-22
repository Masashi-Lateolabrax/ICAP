import os.path

from .prerulde import Settings

from .brain import Brain
from .task import TaskGenerator


def train(log_path):
    from libs.optimizer import CMAES, Hist

    logger = Hist(os.path.dirname(log_path))
    cmaes = CMAES(
        dim=Brain.get_dim(),
        generation=Settings.CMAES_GENERATION,
        population=Settings.CMAES_POPULATION,
        mu=Settings.CMAES_MU,
        sigma=Settings.CMAES_SIGMA,
        logger=logger
    )
    for gen in range(1, 1 + cmaes.get_generation()):
        task_generator = TaskGenerator()
        cmaes.optimize_current_generation(task_generator)

    logger.save(os.path.basename(log_path))
