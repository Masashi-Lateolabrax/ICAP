import numpy as np

from framework.optimization import Loss

from .robot import ROBOT_SIZE
from .food import FOOD_SIZE


class Settings:
    class Simulation:
        TIME_STEP = 0.01
        TIME_LENGTH: int = int(60 / 0.01)  # Episode length (in steps)

        WORLD_WIDTH = 10  # World size
        WORLD_HEIGHT = 10

        class Render:
            RENDER_WIDTH = 300
            RENDER_HEIGHT = 300
            RENDER_ZOOM = 16

            MAX_GEOM = 100  # Maximum number of geometries to render

            LIGHT_AMBIENT = (0.1, 0.1, 0.1)
            LIGHT_DIFFUSE = (0.4, 0.4, 0.4)
            LIGHT_SPECULAR = (0.5, 0.5, 0.5)

    class CMAES:
        GENERATION = 3  # Total number of generations
        POPULATION = 4  # Population size per generation
        MU = 50  # Number of elite individuals (selected)
        SIGMA = 0.1  # Initial standard deviation
        LOSS: Loss = None

        @staticmethod
        def _loss(para: np.ndarray, nest_pos: np.ndarray, robot_pos: np.ndarray, food_pos: np.ndarray) -> float:
            if Settings.CMAES.LOSS is None:
                raise NotImplementedError("Loss function is not implemented. Please set Settings.CMAES.LOSS.")
            return Settings.CMAES.LOSS(para, nest_pos, robot_pos, food_pos)

    class Robot:
        NUM = 1
        THINK_INTERVAL = 0.3  # Unit is second
        MAX_SECRETION = 0.423 * 5
        ARGMAX_SELECTION = False
        POSITION: list[tuple[float, float, float]] = []

        class OtherRobotSensor:
            GAIN = (lambda d, v: (1 - v) / (v * d))(2, 0.1)
            OFFSET = ROBOT_SIZE * 2
            TANH_GAIN = 0.3

        class FoodSensor:
            GAIN = (lambda d, v: (1 - v) / (v * d))(4, 0.1)
            OFFSET = FOOD_SIZE + ROBOT_SIZE
            TANH_GAIN = 0.3

    class Food:
        NUM = 1
        POSITION: list[tuple[float, float]] = []
        REPLACEMENT = True

    class Nest:
        POSITION = (0, 0)

    class Pheromone:
        SATURATION_VAPOR = 1
        EVAPORATION = 1
        DIFFUSION = 0.05
        DECREASE = 0.1
