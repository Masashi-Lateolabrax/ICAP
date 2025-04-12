from libs import optimizer

from ..settings import Settings
from ..simulator.objects import BrainBuilder
from ..task_generator import TaskGenerator


def train(settings: Settings, logger: optimizer.Logger, brain_builder: BrainBuilder):
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
        # cmaes.optimize_current_generation(task_generator, proc=optimizer.OneThreadProc)
