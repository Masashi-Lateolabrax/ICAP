import numpy as np


def _calc_loss_sigma(point, value):
    return np.sqrt(-(point ** 2) / np.log(value))


class _Settings:
    CMAES_GENERATION = 500
    CMAES_POPULATION = 1000
    CMAES_MU = 500
    CMAES_SIGMA = 0.1

    SIMULATION_TIMESTEP = 0.01
    SIMULATION_TIME_LENGTH = 2000

    LOSS_RATE_FN = 1
    LOSS_SIGMA_FN = _calc_loss_sigma(3, 0.1)
    LOSS_RATE_FR = 0.001
    LOSS_SIGMA_FR = _calc_loss_sigma(1, 0.5)

    RENDER_WIDTH = 500
    RENDER_HEIGHT = 500
    RENDER_ZOOM = 11
    MAX_GEOM = 3000

    WORLD_WIDTH = 8
    WORLD_HEIGHT = 8
    WALL_PADDING = 1

    ROBOT_SIZE = 0.175
    ROBOT_WEIGHT = 30  # kg
    ROBOT_MOVE_SPEED = 0.8
    ROBOT_TURN_SPEED = 3.14 / 2
    ROBOT_THINK_INTERVAL = 100

    FOOD_SIZE = 0.5
    FOOD_DENSITY = 300
    FOOD_FRICTIONLOSS = 1.5
    FOOD_NUM = 1

    NEST_POS = (0, 0)
    NEST_SIZE = 0.5

    SENSOR_GAIN = 1

    @property
    def SENSOR_OFFSET(self):
        return self.ROBOT_SIZE + self.FOOD_SIZE


Settings = _Settings()
