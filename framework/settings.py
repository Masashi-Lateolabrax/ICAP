import numpy as np

from .simulator.objects.robot.const import ROBOT_SIZE
from .simulator.objects.food.const import FOOD_SIZE
from .optimization import Loss


class Settings:
    class Simulation:
        TIME_STEP = 0.01
        TIME_LENGTH: int = int(60 / 0.01)  # Episode length (in steps)

        RENDER_WIDTH = 300
        RENDER_HEIGHT = 300
        RENDER_ZOOM = 16
        MAX_GEOM = 100  # Maximum number of geometries to render

        WORLD_WIDTH = 10  # World size
        WORLD_HEIGHT = 10

    class CMAES:
        GENERATION = 3  # Total number of generations
        POPULATION = 4  # Population size per generation
        MU = 50  # Number of elite individuals (selected)
        SIGMA = 0.1  # Initial standard deviation
        LOSS: Loss = None

        @staticmethod
        def _loss(nest_pos: np.ndarray, robot_pos: np.ndarray, food_pos: np.ndarray) -> float:
            if Settings.CMAES.LOSS is None:
                raise NotImplementedError("Loss function is not implemented. Please set Settings.CMAES.LOSS.")
            return Settings.CMAES.LOSS(nest_pos, robot_pos, food_pos)

    class Robot:
        NUM = 1
        THINK_INTERVAL = 0.3  # Unit is second
        MAX_SECRETION = 0.423 * 5
        ARGMAX_SELECTION = False
        POSITION: list[tuple[float, float, float]] = []

        class OtherRobotSensor:
            GAIN = (lambda d, v: (1 - v) / (v * d))(2, 0.1)
            OFFSET = ROBOT_SIZE * 2

        class FoodSensor:
            GAIN = (lambda d, v: (1 - v) / (v * d))(4, 0.1)
            OFFSET = FOOD_SIZE + ROBOT_SIZE

    class Food:
        NUM = 1
        POSITION: list[tuple[float, float]] = []

    class Nest:
        POSITION = (0, 0)

    class Pheromone:
        SATURATION_VAPOR = 1
        EVAPORATION = 1
        DIFFUSION = 0.05
        DECREASE = 0.1
