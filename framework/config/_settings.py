from typing import Callable
import logging
import math

import jax.numpy as jnp

from ..types import RobotLocation, Position, Material, ETHANOL


class ClippingFunctions:
    SIN_AND_EXP_PARMS: dict[str, float] = {"sigma": 10, "gain": 1.0}

    @staticmethod
    def none(x: jnp.ndarray) -> jnp.ndarray:
        return x

    @staticmethod
    def sin_and_exp(x: jnp.ndarray) -> jnp.ndarray:
        gain = ClippingFunctions.SIN_AND_EXP_PARMS["gain"]
        sigma = ClippingFunctions.SIN_AND_EXP_PARMS["sigma"]
        return jnp.sin(x) * jnp.exp(-(x ** 2) / (sigma ** 2)) * gain

    @staticmethod
    def hard_clip(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x, -1.0, 1.0)


def calc_loss_sigma(point: float, value: float) -> float:
    return -(point ** 2) / math.log(value)


class Render:
    RENDER_WIDTH: int = 500
    RENDER_HEIGHT: int = 500

    LIGHT_AMBIENT: float = 1.0
    LIGHT_DIFFUSE: float = 1.0
    LIGHT_SPECULAR: float = 1.0

    CAMERA_POS: tuple[float, float, float] = (0.0, -1e-3, 13.0)
    CAMERA_LOOKAT: tuple[float, float, float] = (0.0, 0.0, 0.0)

    MAX_GEOM: int = 11000
    MAX_PHEROMONE: float = 1.0


class Optimization:
    DIMENSION: int | None = None
    POPULATION: int = 1000
    GENERATION: int = 100
    SIGMA: float = 0.5
    CLIP: Callable[[jnp.ndarray], jnp.ndarray] = ClippingFunctions.none


class Robot:
    HEIGHT: float = 0.1
    RADIUS: float = 0.175
    DISTANCE_BETWEEN_WHEELS: float = 0.175 * 2 * 0.8
    MAX_SPEED: float = 0.8
    MASS: int = 10

    COLOR: tuple[float, float, float, float] = (1, 1, 0, 1)

    THINK_INTERVAL: float = 0.05

    ACTUATOR_MOVE_KV: int = 100
    ACTUATOR_ROT_KV: int = 10

    NUM_RAYS = 16

    ROBOT_SENSOR_GAIN: float = 1.0
    FOOD_SENSOR_GAIN: float = 1.0

    NUM: int = 1
    INITIAL_POSITION: list[RobotLocation] = []


class Food:
    RADIUS: float = 0.5
    HEIGHT: float = 0.07

    DENSITY: int = 80
    COLOR: tuple[float, float, float, float] = (0, 1, 1, 1)

    NUM: int = 1
    INITIAL_POSITION: list[RobotLocation] = []


class Nest:
    POSITION: Position = Position(0.0, 0.0)
    RADIUS: float = 1.0
    HEIGHT: float = 0.01
    COLOR: tuple[float, float, float, float] = (0, 1, 0, 1)


class Loss:
    OFFSET_FOOD_AND_NEST: float = 0
    SIGMA_FOOD_AND_NEST: float = calc_loss_sigma(4, 0.01)
    GAIN_FOOD_AND_NEST: int = 1

    OFFSET_FOOD_AND_ROBOT: float = Robot.RADIUS + Food.RADIUS
    SIGMA_FOOD_AND_ROBOT: float = calc_loss_sigma(1, 0.3)
    GAIN_FOOD_AND_ROBOT: float = 0.01

    REGULARIZATION_COEFFICIENT: int = 0


class Simulation:
    TIME_STEP: float = 0.01
    TIME_LENGTH: int = 60  # Unit is Seconds

    WORLD_WIDTH: float = 10.0
    WORLD_HEIGHT: float = 10.0

    WALL_THICKNESS: float = 1
    WALL_HEIGHT: float = 1

    TEMPERATURE: float = 300.0  # Kelvin


class Storage:
    SAVE_INDIVIDUALS: bool = True
    SAVE_DIRECTORY: str = "./results"
    SAVE_INTERVAL: int = 10  # Save every N generations
    TOP_N: int = 0  # Save top N individuals, 0 means save all
    ASSET_DIRECTORY: str = "./assets"


class Device:
    ENABLE_CUDA: bool = False


class Pheromone:
    ACTIVE: bool = False
    CELL_SIZE: float = 0.1
    WIDTH_NUM: int = int(Simulation.WORLD_WIDTH / CELL_SIZE)
    HEIGHT_NUM: int = int(Simulation.WORLD_HEIGHT / CELL_SIZE)
    ITERATIONS_PER_STEP: int = 1
    EVAPORATION_RATE: float = 0.1
    DECREASE_RATE: float = 0.0
    MATERIAL: Material = ETHANOL


class Settings:
    Optimization: type[Optimization] = Optimization
    Loss: type[Loss] = Loss
    Simulation: type[Simulation] = Simulation
    Render: type[Render] = Render
    Robot: type[Robot] = Robot
    Food: type[Food] = Food
    Nest: type[Nest] = Nest
    Storage: type[Storage] = Storage
    Device: type[Device] = Device
    Pheromone: type[Pheromone] = Pheromone

    @staticmethod
    def as_dict(this: type['Settings']) -> dict:
        def as_dict(obj):
            ALLOWED_TYPES = (str, int, float, bool, Callable, Material, Position, RobotLocation)
            attributes = {}

            for attr_name in dir(obj):
                if attr_name.startswith('_') or attr_name == 'as_dict' or attr_name == 'compare_settings':
                    continue

                attr_value = getattr(obj, attr_name)

                if isinstance(attr_value, type):
                    res = as_dict(attr_value)
                    if res is not None:
                        attributes[attr_name] = res

                elif isinstance(attr_value, ALLOWED_TYPES):
                    attributes[attr_name] = attr_value

                elif isinstance(attr_value, (tuple, list)):
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

        return as_dict(this)

    @staticmethod
    def compare_settings(this: type["Settings"], other: type["Settings"]):
        base_attrs = Settings.as_dict(this)
        app_attrs = Settings.as_dict(other)

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
