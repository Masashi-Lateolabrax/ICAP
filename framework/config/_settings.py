import numpy as np

from ..types import RobotLocation, Position


def calc_loss_sigma(point, value):
    return -(point ** 2) / np.log(value)


class Robot:
    NUM = 1
    INITIAL_POSITION: list[RobotLocation] = []

    HEIGHT = 0.1
    RADIUS = 0.175
    DISTANCE_BETWEEN_WHEELS = 0.175 * 2 * 0.8

    MAX_SPEED = 0.8

    COLOR = (1, 1, 0, 1)
    MASS = 10

    ACTUATOR_MOVE_KV = 100
    ACTUATOR_ROT_KV = 10

    ROBOT_SENSOR_GAIN = 1.0
    FOOD_SENSOR_GAIN = 1.0


class Food:
    NUM: int = 1
    INITIAL_POSITION: list[Position] = [Position(0.0, 0.0)]
    RADIUS = 0.5
    DENSITY = 80
    HEIGHT = 0.07
    COLOR = (0, 1, 1, 1)


class Nest:
    POSITION: Position = Position(0.0, 0.0)
    RADIUS = 1.0
    HEIGHT = 0.01
    COLOR = (0, 1, 0, 1)


class Server:
    HOST: str = 'localhost'
    PORT: int = 5000
    SOCKET_BACKLOG: int = 10
    TIMEOUT: float = 5.0  # seconds


class Optimization:
    DIMENSION: int = 10
    POPULATION: int = 1000
    GENERATION: int = 100
    SIGMA: float = 0.5


class Loss:
    OFFSET_NEST_AND_FOOD = 0
    SIGMA_NEST_AND_FOOD = calc_loss_sigma(4, 0.01)
    GAIN_NEST_AND_FOOD = 1

    OFFSET_ROBOT_AND_FOOD = Robot.RADIUS + Food.RADIUS
    SIGMA_ROBOT_AND_FOOD = calc_loss_sigma(1, 0.3)
    GAIN_ROBOT_AND_FOOD = 0.01

    REGULARIZATION_COEFFICIENT = 0


class Simulation:
    TIME_STEP: float = 0.01
    TIME_LENGTH: int = 60  # Unit is Seconds

    WORLD_WIDTH: float = 10.0
    WORLD_HEIGHT: float = 10.0
    WALL_THICKNESS: float = 1
    WALL_HEIGHT: float = 1


class Render:
    RENDER_WIDTH = 100
    RENDER_HEIGHT = 100
    LIGHT_AMBIENT = 1.0
    LIGHT_DIFFUSE = 1.0
    LIGHT_SPECULAR = 1.0


class Storage:
    SAVE_INDIVIDUALS = True
    SAVE_DIRECTORY = "results"
    SAVE_INTERVAL = 10  # Save every N generations
    TOP_N = 0  # Save top N individuals, 0 means save all


class Settings:
    """
    Basically, the attributes' unit is meter.
    """

    Server = Server
    Optimization = Optimization
    Loss = Loss
    Simulation = Simulation
    Render = Render
    Robot = Robot
    Food = Food
    Nest = Nest
    Storage = Storage
