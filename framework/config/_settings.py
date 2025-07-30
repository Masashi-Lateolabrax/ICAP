import logging

import numpy as np

from ..types import RobotLocation, Position


def calc_loss_sigma(point, value):
    return -(point ** 2) / np.log(value)


class Server:
    HOST: str = 'localhost'
    PORT: int = 5000
    SOCKET_BACKLOG: int = 10


class Render:
    RENDER_WIDTH = 500
    RENDER_HEIGHT = 500

    LIGHT_AMBIENT = 1.0
    LIGHT_DIFFUSE = 1.0
    LIGHT_SPECULAR = 1.0

    CAMERA_POS = (0.0, -1e-3, 13.0)
    CAMERA_LOOKAT = (0.0, 0.0, 0.0)


class Optimization:
    DIMENSION: int = None
    POPULATION: int = 1000
    GENERATION: int = 100
    SIGMA: float = 0.5
    CLIP: tuple[float, float] | None = None  # CLIP[0] is the lower bound, CLIP[1] is the upper bound.


class Robot:
    HEIGHT = 0.1
    RADIUS = 0.175
    DISTANCE_BETWEEN_WHEELS = 0.175 * 2 * 0.8
    MAX_SPEED = 0.8
    MASS = 10

    COLOR = (1, 1, 0, 1)

    THINK_INTERVAL = 0.05

    ACTUATOR_MOVE_KV = 100
    ACTUATOR_ROT_KV = 10

    ROBOT_SENSOR_GAIN = 1.0
    FOOD_SENSOR_GAIN = 1.0

    NUM = 1
    INITIAL_POSITION: list[RobotLocation] = []


class Food:
    RADIUS = 0.5
    HEIGHT = 0.07

    DENSITY = 80
    COLOR = (0, 1, 1, 1)

    NUM: int = 1
    INITIAL_POSITION: list[Position] = [Position(0.0, 0.0)]


class Nest:
    POSITION: Position = Position(0.0, 0.0)
    RADIUS = 1.0
    HEIGHT = 0.01
    COLOR = (0, 1, 0, 1)


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


class Storage:
    SAVE_INDIVIDUALS = True
    SAVE_DIRECTORY = "./results"
    SAVE_INTERVAL = 10  # Save every N generations
    TOP_N = 0  # Save top N individuals, 0 means save all
    ASSET_DIRECTORY = "./assets"


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

    def as_dict(self):
        def as_dict(obj):
            ALLOWED_TYPES = (str, int, float, bool, Position, RobotLocation)
            attributes = {}

            for attr_name in dir(obj):
                if attr_name.startswith('_') or attr_name == 'as_dict':
                    continue

                attr_value = getattr(obj, attr_name)

                if isinstance(attr_value, type):
                    res = as_dict(attr_value)
                    if res is not None:
                        attributes[attr_name] = res

                elif isinstance(attr_value, ALLOWED_TYPES):
                    attributes[attr_name] = attr_value

                elif isinstance(attr_value, tuple):
                    tuple_ = []
                    for v in attr_value:
                        if not isinstance(v, ALLOWED_TYPES):
                            logging.warning("Skipping non-serializable value in tuple: %s", v)
                            continue
                        tuple_.append(v)
                    attributes[attr_name] = tuple(tuple_)

                elif isinstance(attr_value, list):
                    list_ = []
                    for v in attr_value:
                        if not isinstance(v, ALLOWED_TYPES):
                            logging.warning("Skipping non-serializable value in list: %s", v)
                            continue
                        list_.append(v)
                    attributes[attr_name] = list_

                elif isinstance(attr_value, dict):
                    dict_ = {}
                    for k, v in attr_value.items():
                        if not isinstance(k, (str, int)) or not isinstance(v, ALLOWED_TYPES):
                            logging.warning("Skipping non-serializable key-value pair in dict: %s: %s", k, v)
                            continue
                        dict_[k] = v
                    attributes[attr_name] = dict_

            return attributes

        return as_dict(self)

    def compare_settings(self, other: 'Settings'):
        base_attrs = self.as_dict()
        app_attrs = other.as_dict()

        # Find all unique keys
        all_keys = set(base_attrs.keys()) | set(app_attrs.keys())

        differences = []
        identical = []

        for key in sorted(all_keys):
            base_value = base_attrs.get(key, "<NOT SET>")
            app_value = app_attrs.get(key, "<NOT SET>")

            if base_value != app_value:
                differences.append((key, base_value, app_value))
            else:
                identical.append((key, base_value))

        # Print differences
        if differences:
            print(f"\n🔍 DIFFERENCES FOUND ({len(differences)} settings):")
            print("-" * 60)
            for key, base_val, app_val in differences:
                print(f"Setting: {key}")
                print(f"  self: {base_val}")
                print(f"  other: {app_val}")
                print()

        return {"difference": differences, "identical": identical}
