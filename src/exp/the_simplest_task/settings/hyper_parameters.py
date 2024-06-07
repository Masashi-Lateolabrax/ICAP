class HyperParameters:
    class Optimization:
        GENERATION = 1000
        POPULATION = 15
        SIGMA = 0.3
        EARLY_STOP = 0.01

    class Evaluation:
        ANGLE_WEIGHT = 15

    class Simulator:
        MAX_GEOM = 1000
        EPISODE = 30
        TIMESTEP = 0.01
        RESOLUTION = (900, 1350)
        TRY_COUNT = 20
        COLOR_MAP = {
            "bot": (255, 255, 0),
            "goal": (0, 255, 0),
        }

    class Environment:
        GOAL_POS = (0, 5)
        BOT_POS = (0, -5)

    class Robot:
        NUM_LIDAR = 314
        SENSOR_PRECISION = 0.05
        SIGHT_KERNEL_SIZE = 40

        MOVE_SPEED = 1.2
        TURN_SPEED = 1.0


class StaticParameters:
    @staticmethod
    def input_size():
        k = HyperParameters.Robot.SIGHT_KERNEL_SIZE
        i = HyperParameters.Robot.NUM_LIDAR
        return int((i + 2 * int(k * 0.5 + 0.5) - k) / int(k * 0.5) + 1)

    @staticmethod
    def distance_measure_gain(dist: float) -> float:
        return 1 / (dist * HyperParameters.Robot.SENSOR_PRECISION + 1)
