import logging
from typing import List, Optional

import numpy as np

from ..types import RobotLocation, Position


def calc_loss_sigma(point, value):
    return -(point ** 2) / np.log(value)


class Server:
    def __init__(self):
        self.HOST: str = 'localhost'
        self.PORT: int = 5000
        self.SOCKET_BACKLOG: int = 10


class Render:
    def __init__(self):
        self.RENDER_WIDTH: int = 500
        self.RENDER_HEIGHT: int = 500

        self.LIGHT_AMBIENT: float = 1.0
        self.LIGHT_DIFFUSE: float = 1.0
        self.LIGHT_SPECULAR: float = 1.0

        self.CAMERA_POS: tuple = (0.0, -1e-3, 13.0)
        self.CAMERA_LOOKAT: tuple = (0.0, 0.0, 0.0)


class Optimization:
    def __init__(self):
        self.DIMENSION: Optional[int] = None
        self.POPULATION: int = 1000
        self.GENERATION: int = 100
        self.SIGMA: float = 0.5


class Robot:
    def __init__(self):
        self.HEIGHT: float = 0.1
        self.RADIUS: float = 0.175
        self.DISTANCE_BETWEEN_WHEELS: float = 0.175 * 2 * 0.8
        self.MAX_SPEED: float = 0.8
        self.MASS: float = 10

        self.COLOR: tuple = (1, 1, 0, 1)

        self.THINK_INTERVAL: float = 0.05

        self.ACTUATOR_MOVE_KV: float = 100
        self.ACTUATOR_ROT_KV: float = 10

        self.ROBOT_SENSOR_GAIN: float = 1.0
        self.FOOD_SENSOR_GAIN: float = 1.0

        self.NUM: int = 1
        self.INITIAL_POSITION: List[RobotLocation] = []


class Food:
    def __init__(self):
        self.RADIUS: float = 0.5
        self.HEIGHT: float = 0.07

        self.DENSITY: float = 80
        self.COLOR: tuple = (0, 1, 1, 1)

        self.NUM: int = 1
        self.INITIAL_POSITION: List[Position] = [Position(0.0, 0.0)]


class Nest:
    def __init__(self):
        self.POSITION: Position = Position(0.0, 0.0)
        self.RADIUS: float = 1.0
        self.HEIGHT: float = 0.01
        self.COLOR: tuple = (0, 1, 0, 1)


class Loss:
    def __init__(self):
        self.OFFSET_NEST_AND_FOOD: float = 0
        self.SIGMA_NEST_AND_FOOD: float = calc_loss_sigma(4, 0.01)
        self.GAIN_NEST_AND_FOOD: float = 1

        self.OFFSET_ROBOT_AND_FOOD: float = 0.175 + 0.5  # Will be set properly in Settings
        self.SIGMA_ROBOT_AND_FOOD: float = calc_loss_sigma(1, 0.3)
        self.GAIN_ROBOT_AND_FOOD: float = 0.01
        self.REGULARIZATION_COEFFICIENT: float = 0


class Simulation:
    def __init__(self):
        self.TIME_STEP: float = 0.01
        self.TIME_LENGTH: int = 60  # Unit is Seconds

        self.WORLD_WIDTH: float = 10.0
        self.WORLD_HEIGHT: float = 10.0

        self.WALL_THICKNESS: float = 1
        self.WALL_HEIGHT: float = 1


class Storage:
    def __init__(self):
        self.SAVE_INDIVIDUALS: bool = True
        self.SAVE_DIRECTORY: str = "./results"
        self.SAVE_INTERVAL: int = 10  # Save every N generations
        self.TOP_N: int = 0  # Save top N individuals, 0 means save all
        self.ASSET_DIRECTORY: str = "./assets"


class Settings:
    """
    Basically, the attributes' unit is meter.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.Server = Server()
        self.Optimization = Optimization()
        self.Robot = Robot()
        self.Food = Food()
        self.Nest = Nest()
        self.Loss = Loss()
        self.Simulation = Simulation()
        self.Render = Render()
        self.Storage = Storage()

        # Fix Loss dependencies after instances are created
        self.Loss.OFFSET_ROBOT_AND_FOOD = self.Robot.RADIUS + self.Food.RADIUS
        self._initialized = True

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
