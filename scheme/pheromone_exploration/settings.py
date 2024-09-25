class _HasPropertyMethod:
    def __init__(self, parent):
        self.parent = parent


class Settings:
    NUM_GENERATION = 10

    class Display:
        MAX_GEOM = 7800
        RESOLUTION = (500, 500)
        ZOOM = 13

    class Plot:
        START = 0
        END = 5
        AT_POINT = 0.35

    @property
    class Simulation(_HasPropertyMethod):
        EPISODE_LENGTH = 5
        TIMESTEP = 0.005
        EPS = 1
        MIN_SV = 0.1

        @property
        def TOTAL_STEP(self):
            return int(self.EPISODE_LENGTH / self.TIMESTEP + 0.5)

    @property
    class Environment(_HasPropertyMethod):
        NUM_CELL = (51, 51)
        CELL_SIZE = 0.2
        LIQUID = 1000000000

        @property
        def CENTER_INDEX(self):
            return int(self.NUM_CELL[0] * 0.5), int(self.NUM_CELL[1] * 0.5)

    class Pheromone:
        SATURATION_VAPOR = 10
        EVAPORATION = 10
        DIFFUSION = 10
        DECREASE = 10
