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

    backend = Simulator(settings, individual)
    for _ in range(math.ceil(settings.Simulation.TIME_LENGTH / settings.Simulation.TIME_STEP)):
        backend.step()

    return backend.total_score()


