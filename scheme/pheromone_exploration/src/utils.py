import random

from .settings import Settings


def gen_parameters():
    return {
        "sv": Settings.Pheromone.SATURATION_VAPOR * random.random(),
        "evaporate": Settings.Pheromone.EVAPORATION * random.random(),
        "diffusion": Settings.Pheromone.DIFFUSION * random.random(),
        "decrease": Settings.Pheromone.DECREASE * random.random()
    }
