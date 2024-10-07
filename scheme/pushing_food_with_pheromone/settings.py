import random


class _PropertyClass:
    def __init__(self, parent):
        self.parent = parent


CACHE_NUM_ROBOT = 0
CACHE_NUM_FOOD = 0


class Settings:
    @property
    class Optimization(_PropertyClass):
        GENERATION = 500
        POPULATION = 50  # int(4 + 3 * math.log(211))
        SIGMA = 0.3

        @property
        def MU(self):
            return int(self.POPULATION * 0.5 + 0.5)

        @property
        class Evaluation(_PropertyClass):
            FOOD_GAIN = 1
            FOOD_RANGE = 6
            FOOD_RANGE_P = 2

            NEST_GAIN = 1
            NEST_RANGE_P = 2

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
                pos = [[0, -6], [0, 6]]
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

    class Simulation:
        TIMESTEP = 0.01

        class Pheromone:
            TIMESTEP = 0.01

    class Renderer:
        MAX_GEOM = 7800
        RESOLUTION = [900, 1350]

    @property
    class Characteristic(_PropertyClass):
        class Environment:
            CELL_SIZE = 0.2
            WIDTH = 80
            HEIGHT = 90

        @property
        class Robot(_PropertyClass):
            THINKING_INTERVAL = 0.3

            MOVE_SPEED = 1.2
            TURN_SPEED = 1.0

            SENSOR_PRECISION = [2 / 0.7, 2 / 5]

            MAX_SECRETION = 0.423 * 3

            @property
            def TANK_SIZE(self):
                return 3 * self.MAX_SECRETION

        class Pheromone:
            SATURATION_VAPOR = 1
            EVAPORATION = 1
            DIFFUSION = 0.05
            DECREASE = 0.1
