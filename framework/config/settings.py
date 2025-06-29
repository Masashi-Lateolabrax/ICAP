import dataclasses


@dataclasses.dataclass
class Position:
    x: float
    y: float


class RobotLocation:
    def __init__(self, x: float, y: float, theta: float):
        self.position = Position(x, y)
        self.theta = theta  # Unit is degree

    @property
    def x(self) -> float:
        return self.position.x

    @property
    def y(self) -> float:
        return self.position.y


class Settings:
    """
    Basically, the attributes' unit is meter.
    """

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
        HEIGHT = 0.05
        RADIUS = 0.175
        COLOR = (1, 1, 0, 1)
        MASS = 10
        ACTUATOR_KV = 0.5

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
