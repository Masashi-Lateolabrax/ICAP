cx, cy = (0, 0)


class HyperParameters:
    class Optimization:
        GENERATION = 500
        POPULATION = 100  # int(4 + 3 * math.log(211))
        MU = 10
        SIGMA = 0.3

    class Simulator:
        MAX_GEOM = 7800
        EPISODE = 30
        TIMESTEP = 0.01
        RESOLUTION = (900, 1350)
        TILE_SIZE = 0.2
        TILE_D = 1
        TILE_WH = (80, 90)
        PHEROMONE_ITER = 5

    class Evaluation:
        FOOD_RANGE = 2.236067977
        FOOD_NEST_GAIN = 1
        FOOD_ROBOT_GAIN = 1e-3

    class Pheromone:
        SaturatedVapor = 1.50
        Evaporation = 14.9
        Diffusion = 6.50
        Decrease = 0.00186

    class Environment:
        NEST_POS = (cx, cy)
        NEST_SIZE = 0.5

        BOT_POS = [
            (cx - 0.45, cy + 0.45, 90), (cx + 0, cy + 0.45, 90), (cx + 0.45, cy + 0.45, 90),
            (cx - 0.45, cy + 0.0, 90), (cx + 0, cy + 0.0, 90), (cx + 0.45, cy + 0.0, 90),
            (cx - 0.45, cy - 0.45, 90), (cx + 0, cy - 0.45, 90), (cx + 0.45, cy - 0.45, 90),
        ]

        FOOD_POS = [
            (0, -5), (0, 5)
        ]

    class Robot:
        MAX_SECRETION = 2.569317817
        SENSOR_PRECISION = (2 / 0.7, 2 / 5)
        MOVE_SPEED = 1.2
        TURN_SPEED = 1.0

# print(f"population: {HyperParameters.Optimization.POPULATION}")
