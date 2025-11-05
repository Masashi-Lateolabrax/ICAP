"""
Base patterns for robot differential drive actions.

This module contains only the fundamental building blocks for action construction.
Each pattern is [right_wheel, left_wheel, pheromone].
"""

import numpy as np

FORWARD_PATTERN = np.array([1.0, 1.0, 0.0])
BACKWARD_PATTERN = np.array([-1.0, -1.0, 0.0])

TURN_LEFT = np.array([1.0, -1.0, 0.0])
TURN_RIGHT = np.array([-1.0, 1.0, 0.0])

SECRETE_PHEROMONE = np.array([0.0, 0.0, 1.0])
