import logging

import numpy as np

from ..types import RobotLocation, Position


def calc_loss_sigma(point, value):
    return -(point ** 2) / np.log(value)


class Server:
    def __init__(self):
        self.HOST = 'localhost'
        self.PORT = 5000
        self.SOCKET_BACKLOG = 10


class Render:
    def __init__(self):
        self.RENDER_WIDTH = 500
        self.RENDER_HEIGHT = 500

        self.LIGHT_AMBIENT = 1.0
        self.LIGHT_DIFFUSE = 1.0
        self.LIGHT_SPECULAR = 1.0

        self.CAMERA_POS = (0.0, -1e-3, 13.0)
        self.CAMERA_LOOKAT = (0.0, 0.0, 0.0)


class Optimization:
    def __init__(self):
        self.DIMENSION = None
        self.POPULATION = 1000
        self.GENERATION = 100
        self.SIGMA = 0.5


class Robot:
    def __init__(self):
        self.HEIGHT = 0.1
        self.RADIUS = 0.175
        self.DISTANCE_BETWEEN_WHEELS = 0.175 * 2 * 0.8
        self.MAX_SPEED = 0.8
        self.MASS = 10

        self.COLOR = (1, 1, 0, 1)

        self.THINK_INTERVAL = 0.05

        self.ACTUATOR_MOVE_KV = 100
        self.ACTUATOR_ROT_KV = 10

        self.ROBOT_SENSOR_GAIN = 1.0
        self.FOOD_SENSOR_GAIN = 1.0

        self.NUM = 1
        self.INITIAL_POSITION = []


class Food:
    def __init__(self):
        self.RADIUS = 0.5
        self.HEIGHT = 0.07

        self.DENSITY = 80
        self.COLOR = (0, 1, 1, 1)

        self.NUM = 1
        self.INITIAL_POSITION = [Position(0.0, 0.0)]


class Nest:
    def __init__(self):
        self.POSITION = Position(0.0, 0.0)
        self.RADIUS = 1.0
        self.HEIGHT = 0.01
        self.COLOR = (0, 1, 0, 1)


class Loss:
    def __init__(self):
        self.OFFSET_NEST_AND_FOOD = 0
        self.SIGMA_NEST_AND_FOOD = calc_loss_sigma(4, 0.01)
        self.GAIN_NEST_AND_FOOD = 1

        self.OFFSET_ROBOT_AND_FOOD = 0.175 + 0.5  # Will be set properly in Settings
        self.SIGMA_ROBOT_AND_FOOD = calc_loss_sigma(1, 0.3)
        self.GAIN_ROBOT_AND_FOOD = 0.01
        self.REGULARIZATION_COEFFICIENT = 0


class Simulation:
    def __init__(self):
        self.TIME_STEP = 0.01
        self.TIME_LENGTH = 60  # Unit is Seconds

        self.WORLD_WIDTH = 10.0
        self.WORLD_HEIGHT = 10.0

        self.WALL_THICKNESS = 1
        self.WALL_HEIGHT = 1


class Storage:
    def __init__(self):
        self.SAVE_INDIVIDUALS = True
        self.SAVE_DIRECTORY = "./results"
        self.SAVE_INTERVAL = 10  # Save every N generations
        self.TOP_N = 0  # Save top N individuals, 0 means save all
        self.ASSET_DIRECTORY = "./assets"


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
