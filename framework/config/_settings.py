from ..types import RobotLocation, Position


class Server:
    HOST: str = 'localhost'
    PORT: int = 5000
    MAX_CONNECTIONS: int = 10
    TIMEOUT: float = 5.0  # seconds


class Optimization:
    dimension: int = 10
    population_size: int = 20
    generations: int = 1000
    sigma: float = 0.5


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


class Settings:
    """
    Basically, the attributes' unit is meter.
    """

    Server = Server
    Optimization = Optimization
    Simulation = Simulation
    Render = Render
    Robot = Robot
    Food = Food
    Nest = Nest
