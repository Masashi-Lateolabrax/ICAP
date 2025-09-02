import dataclasses

import numpy as np


@dataclasses.dataclass
class DebugData:
    time: float
    robot_positions: list[np.ndarray]
    robot_inputs: np.ndarray
    robot_directions: list[np.ndarray]
    food_positions: list[np.ndarray]
    food_directions: list[np.ndarray]
