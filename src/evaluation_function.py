"""
Shared evaluation function for optimization clients.

This module provides the common evaluation logic used by both
main.py and main_parallel.py clients.
"""

import math
from framework.prelude import *
from settings import MySettings
from simulator import Simulator


def evaluation_function(individual: Individual):
    """
    Evaluate an individual's fitness using the MuJoCo simulation.
    
    Args:
        individual: Individual to evaluate
        
    Returns:
        float: Fitness score from simulation
    """
    settings = MySettings()

    RobotValues.set_max_speed(settings.Robot.MAX_SPEED)
    RobotValues.set_distance_between_wheels(settings.Robot.DISTANCE_BETWEEN_WHEELS)
    RobotValues.set_robot_height(settings.Robot.HEIGHT)

    backend = Simulator(settings, individual)
    for _ in range(math.ceil(settings.Simulation.TIME_LENGTH / settings.Simulation.TIME_STEP)):
        backend.step()

    return backend.total_score()


