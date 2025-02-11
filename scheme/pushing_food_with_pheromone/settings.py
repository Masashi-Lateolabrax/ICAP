class _PropertyClass:
    def __init__(self, parent):
        self.parent = parent


class Settings:
    @property
    class Optimization(_PropertyClass):
        GENERATION = 3
        POPULATION = 10  # int(4 + 3 * math.log(211))
        SIGMA = 0.3

        @property
        def MU(self):
            return int(self.POPULATION * 0.5 + 0.5)

        class Evaluation:
            FOOD_GAIN = 1
            FOOD_RANGE = 6
            FOOD_RANGE_P = 2

            NEST_GAIN = 7
            NEST_RANGE_P = 2

    @property
    class Task(_PropertyClass):
        EPISODE = 60
        REDISTRIBUTION_MARGIN = 1.0

        class Nest:
            POSITION = [0, 0]
            SIZE = 1.5  # Radius

        class Robot(_PropertyClass):
            NUM_ROBOTS = 9

            THINKING_INTERVAL = 0.1
            TANK_SIZE = 10

            WEIGHT = 30  # kg

            SIZE = 0.175  # radius

            SECRETION = 1

            MOVE_SPEED = 0.8
            TURN_SPEED = 1.0

            SENSOR_GAIN = [2 / 0.7, 2 / 5]
            SENSOR_OFFSET = [2 / 0.7, 2 / 5]

            MAX_SECRETION = 0.423 * 5

        class Pheromone:
            EVAPORATION = 0.1
            DIFFUSION = 0.1
            NEAR = 0.01

        class World(_PropertyClass):
            WIDTH = 80
            HEIGHT = 90

        class Food(_PropertyClass):
            NUM_FOOD = 2
            SIZE = 1.5  # Radius

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
        ZOOM = 29
        PHEROMONE_MAX = 1.0

        class Resolution:
            WIDTH = 900
            HEIGHT = 1350
