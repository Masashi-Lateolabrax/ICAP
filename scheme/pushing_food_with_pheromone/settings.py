import random
from enum import IntEnum


class _PropertyClass:
    def __init__(self, parent):
        self.parent = parent


CACHE_NUM_ROBOT = 0
CACHE_NUM_FOOD = 0


class EType(IntEnum):
    POTENTIAL = 0
    DISTANCE = 1


class Settings:
    @property
    class Optimization(_PropertyClass):
        GENERATION = 1000
        POPULATION = 100  # int(4 + 3 * math.log(211))
        SIGMA = 0.3
        EVALUATION_TYPE = EType.DISTANCE

        @property
        def MU(self):
            return int(self.POPULATION * 0.5 + 0.5)

        @property
        class Evaluation(_PropertyClass):
            FOOD_GAIN = 1
            FOOD_RANGE = 6
            FOOD_RANGE_P = 2

            NEST_GAIN = 7
            NEST_RANGE_P = 2

        class OldEvaluation:
            FOOD_RANGE = 2.3
            FOOD_NEST_GAIN = 5.0
            FOOD_ROBOT_GAIN = 2.756

    @property
    class Task(_PropertyClass):
        EPISODE = 60

        class Nest:
            POSITION = [0, 0]
            SIZE = 1.5

        @property
        class Robot(_PropertyClass):
            def POSITIONS(self, sigma):
                pos = [
                    [-0.45, 0.45], [0, 0.45], [0.45, 0.45],
                    [-0.45, 0.00], [0, 0.00], [0.45, 0.00],
                    [-0.45, -0.45], [0, -0.45], [0.45, -0.45],
                ]
                global CACHE_NUM_ROBOT
                if CACHE_NUM_ROBOT == 0:
                    CACHE_NUM_ROBOT = len(pos)
                return [(p[0], p[1], 90 + sigma * 180 * (2 * random.random() - 1)) for p in pos]

            @property
            def NUM_ROBOTS(self):
                global CACHE_NUM_ROBOT
                if CACHE_NUM_ROBOT == 0:
                    CACHE_NUM_ROBOT = len(self.POSITIONS(0))
                return CACHE_NUM_ROBOT

        @property
        class Food(_PropertyClass):
            def POSITIONS(self):
                pos = [
                    [0, -3 - 4 * random.random()],
                    [0, 3 + 4 * random.random()]
                ]
                global CACHE_NUM_FOOD
                if CACHE_NUM_FOOD == 0:
                    CACHE_NUM_FOOD = len(pos)
                return pos

            @property
            def NUM_FOOD(self):
                global CACHE_NUM_FOOD
                if CACHE_NUM_FOOD == 0:
                    CACHE_NUM_FOOD = len(self.POSITIONS())
                return CACHE_NUM_FOOD

    @property
    class Simulation(_PropertyClass):
        TIMESTEP = 0.01

        @property
        def TOTAL_STEP(self):
            return int(self.parent.Task.EPISODE / self.TIMESTEP + 0.5)

        class Pheromone:
            TIMESTEP = 0.01

        CEIL_THICKNESS = 1

        @property
        def CEIL_HEIGHT(self):
            return self.parent.Renderer.ZOOM + self.CEIL_THICKNESS + 0.1

    class Renderer:
        MAX_GEOM = 7800
        RESOLUTION = [900, 1350]
        ZOOM = 29

    @property
    class Characteristic(_PropertyClass):
        class Environment:
            CELL_SIZE = 0.2
            WIDTH = 80
            HEIGHT = 90

        @property
        class Robot(_PropertyClass):
            THINKING_INTERVAL = 0.1

            MOVE_SPEED = 1.2
            TURN_SPEED = 1.0

            SENSOR_PRECISION = [2 / 0.7, 2 / 5]

            MAX_SECRETION = 0.423 * 5

            @property
            def TANK_SIZE(self):
                return 3 * self.MAX_SECRETION

        class Pheromone:
            SATURATION_VAPOR = 1
            EVAPORATION = 1
            DIFFUSION = 0.05
            DECREASE = 0.1
