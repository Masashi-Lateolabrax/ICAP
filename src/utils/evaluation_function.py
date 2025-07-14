import math
from typing import Callable

from framework.prelude import Settings, SimulatorBackend, Individual


class EvaluationFunction:
    def __init__(self, settings: Settings, simulator_builder: Callable[[Individual], SimulatorBackend]):
        self.settings = settings
        self.simulator_builder = simulator_builder

    def run(self, individual: Individual):
        backend = self.simulator_builder(individual)
        for _ in range(math.ceil(self.settings.Simulation.TIME_LENGTH / self.settings.Simulation.TIME_STEP)):
            backend.step()
        return backend.total_score()
