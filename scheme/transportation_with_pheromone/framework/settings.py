class __Settings:
    class Simulation:
        TIME_LENGTH = 60  # Episode length (in steps)
        RENDER_WIDTH = 800
        RENDER_HEIGHT = 600
        RENDER_ZOOM = 1.5
        MAX_GEOM = 100  # Maximum number of geometries to render

    class CMAES:
        GENERATION = 1000  # Total number of generations
        POPULATION = 100  # Population size per generation
        MU = 50  # Number of elite individuals (selected)
        SIGMA = 0.5  # Initial standard deviation

    class Robot:
        BRAIN_DIMENSION = 56  # As specified in the task note
        MAX_SECRETION = 0.423 * 5

    class LossFunction:
        FOOD_GAIN = 1
        FOOD_RANGE = 6
        FOOD_RANGE_P = 2
        NEST_GAIN = 7
        NEST_RANGE_P = 2

    class Pheromone:
        SATURATION_VAPOR = 1
        EVAPORATION = 1
        DIFFUSION = 0.05
        DECREASE = 0.1


Settings = __Settings()
