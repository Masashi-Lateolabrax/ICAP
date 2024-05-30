cx, cy = (0, -8)


class HyperParameters:
    class Optimization:
        GENERATION = 1000
        POPULATION = 20
        SIGMA = 0.3

    class Simulator:
        EPISODE = 60
        TIMESTEP = 0.002

    class Evaluation:
        FOOD_RANGE = 5
        FOOD_NEST_GAIN = 1
        FOOD_ROBOT_GAIN = 1

    class Environment:
        NEST_POS = (cx, cy)
        NEST_SIZE = 0.5

        BOT_POS = [
            # (cx + 0, cy + 2.4, 0),
            # (cx - 0.8, cy + 2.4, 120), (cx + 0, cy + 2.4, 180), (cx + 0.8, cy + 2.4, -120),
            (cx - 0.8, cy + 2.4, 0), (cx + 0, cy + 2.4, 0), (cx + 0.8, cy + 2.4, 0),
            # (cx - 0.8, cy + 1.6, 0), (cx + 0, cy + 1.6, 0), (cx + 0.8, cy + 1.6, 0),
            # (cx - 0.8, cy + 0.8, 0), (cx + 0, cy + 0.8, 0), (cx + 0.8, cy + 0.8, 0),
        ]

        FOOD_POS = [
            (0, 1),  # (0, 6)
        ]

    class Robot:
        SENSOR_PRECISION = 0.005
        MOVE_SPEED = 1.2
        TURN_SPEED = 1.0
