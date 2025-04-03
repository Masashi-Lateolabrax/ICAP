import os

from libs import optimizer

from ..settings import Settings
from ..interfaceis import BrainBuilder
from ..task_generator import TaskGenerator


def train(settings: Settings, log_path, brain_builder: BrainBuilder):
    logger = optimizer.Hist(os.path.dirname(log_path))
    cmaes = optimizer.CMAES(
        dim=brain_builder.get_dim(),
        generation=settings.CMAES.GENERATION,
        population=settings.CMAES.POPULATION,
        mu=settings.CMAES.MU,
        sigma=settings.CMAES.SIGMA,
        logger=logger,
    )
    for gen in range(1, 1 + cmaes.get_generation()):
        task_generator = TaskGenerator(settings, brain_builder)
        cmaes.optimize_current_generation(task_generator, proc=optimizer.MultiThreadProc)

    logger.save(os.path.basename(log_path))
