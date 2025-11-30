import dataclasses
import pickle
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class DebugData:
    time: float
    robot_positions: list[np.ndarray]
    robot_inputs: np.ndarray
    robot_outputs: np.ndarray
    robot_directions: list[np.ndarray]
    food_positions: list[np.ndarray]
    food_directions: list[np.ndarray]
    total_gas_pheromone: float
    total_liquid_pheromone: float

    @classmethod
    def load(cls, debug_data_path: Path) -> list['DebugData']:
        """Load list of DebugData from pickle file."""
        with open(debug_data_path, 'rb') as f:
            debug_data = pickle.load(f)
        print(f"Loaded debug data from: {debug_data_path}")
        print(f"  Total frames: {len(debug_data)}")
        return debug_data
