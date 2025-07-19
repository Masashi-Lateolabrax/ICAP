import math
from typing import Callable

from framework.prelude import Settings, SimulatorBackend, Individual


class EvaluationFunction:
    def __init__(self, simulator_builder: Callable[[Individual], SimulatorBackend]):
        self.simulator_builder = simulator_builder

    def run(self, individual: Individual):
        if not hasattr(individual, 'settings') or individual.settings is None:
            raise ValueError("Individual object missing required settings")

        settings = individual.settings
        if not isinstance(settings, Settings):
            raise TypeError("Individual settings must be an instance of Settings")

        backend = self.simulator_builder(individual)
        for _ in range(math.ceil(settings.Simulation.TIME_LENGTH / settings.Simulation.TIME_STEP)):
            backend.step()
        return backend.calc_total_score()
