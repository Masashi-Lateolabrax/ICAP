import copy
from dataclasses import dataclass

import mujoco

from src.config import Simulator


@dataclass
class SimulatorState:
    """Complete simulator state snapshot."""
    data: mujoco.MjData
    timer_time: int
    pheromone_field: any  # Type depends on implementation
    dummy_foods: list
    rng_state: dict

    def __init__(self, simulator: Simulator):
        """Create state snapshot from simulator."""
        data_copy = mujoco.MjData(simulator.model)
        mujoco.mj_copyData(data_copy, simulator.model, simulator.data)

        self.data = data_copy
        self.timer_time = simulator.timer.time
        self.pheromone_field = copy.deepcopy(
            simulator._pheromone_field) if simulator._pheromone_field is not None else None
        self.dummy_foods = copy.deepcopy(simulator.dummy_foods)
        self.rng_state = copy.deepcopy(simulator.rng.bit_generator.state)

    def restore(self, simulator: Simulator):
        """Restore simulator to this saved state."""
        mujoco.mj_copyData(simulator.data, simulator.model, self.data)
        simulator.timer.time = self.timer_time
        simulator._pheromone_field = copy.deepcopy(self.pheromone_field) if self.pheromone_field is not None else None
        simulator.dummy_foods = copy.deepcopy(self.dummy_foods)
        simulator.rng.bit_generator.state = copy.deepcopy(self.rng_state)
