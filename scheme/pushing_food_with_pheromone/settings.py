class _HasPropertyMethod:
    def __init__(self, parent):
        self.parent = parent


class Settings:
    @property
    class Optimization(_HasPropertyMethod):
        GENERATION = 500
        POPULATION = 50  # int(4 + 3 * math.log(211))
        SIGMA = 0.3

        @property
        def MU(self):
            return int(self.POPULATION * 0.5 + 0.5)

    class Evaluation:
        FOOD_RANGE = 2.3
        FOOD_NEST_GAIN = 1
        FOOD_ROBOT_GAIN = 0.01

    class Task:
        EPISODE = 30

        class Nest:
            POSITION = [0, 0]
            SIZE = 3.0

        class Robot:
            POSITIONS = [
                [-0.45, 0.45], [0, 0.45], [0.45, 0.45],
                [-0.45, 0.00], [0, 0.00], [0.45, 0.00],
                [-0.45, -0.45], [0, -0.45], [0.45, -0.45],
            ]

        class Food:
            POSITIONS = [
                [0, -6], [0, 6]
            ]

    class Simulation:
        TIMESTEP = 0.01

        class Pheromone:
            TIMESTEP = 0.0005

    class Renderer:
        MAX_GEOM = 7800
        RESOLUTION = [900, 1350]

    class Characteristic:
        class Environment:
            CELL_SIZE = 0.2
            WIDTH = 80
            HEIGHT = 90

        @property
        class Robot(_HasPropertyMethod):
            THINKING_INTERVAL = 0.3

            MOVE_SPEED = 1.2
            TURN_SPEED = 1.0

            SENSOR_PRECISION = [2 / 0.7, 2 / 5]

            @property
            def MAX_SECRETION(self):
                return 0.423 * 3

            @property
            def TANK_SIZE(self):
                return 3 * self.MAX_SECRETION

        class Pheromone:
            SATURATION_VAPOR = 5.442554913756431
            EVAPORATION = 9.390048012289446
            DIFFUSION = 0.4907347580652921
            DECREASE = 5.136664772909677
