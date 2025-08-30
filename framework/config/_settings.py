from typing import Callable, Any

import jax
import jax.numpy as jnp
from flax.struct import dataclass as jax_dataclass

from ..types import RobotLocation, Position, ETHANOL


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


def calc_loss_sigma(point: float, value: float) -> jnp.ndarray:
    return -(point ** 2) / jnp.log(value)


@jax_dataclass
class Render:
    RENDER_WIDTH: int = 500
    RENDER_HEIGHT: int = 500

    LIGHT_AMBIENT: float = 1.0
    LIGHT_DIFFUSE: float = 1.0
    LIGHT_SPECULAR: float = 1.0

    CAMERA_POS: tuple[float, float, float] = (0.0, -1e-3, 13.0)
    CAMERA_LOOKAT: tuple[float, float, float] = (0.0, 0.0, 0.0)

    MAX_GEOM: int = 11000


@jax_dataclass
class Optimization:
    DIMENSION: int | None = None
    POPULATION: int = 1000
    GENERATION: int = 100
    SIGMA: float = 0.5
    CLIP: Callable[[jnp.ndarray], jnp.ndarray] = ClippingFunctions.none
    
    def __post_init__(self) -> None:
        object.__setattr__(self, 'CLIP', jax.jit(self.CLIP))


@jax_dataclass
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

    ROBOT_SENSOR_GAIN: float = 1.0
    FOOD_SENSOR_GAIN: float = 1.0

    NUM: int = 1
    INITIAL_POSITION: list[RobotLocation] = []


@jax_dataclass
class Food:
    RADIUS: float = 0.5
    HEIGHT: float = 0.07

    DENSITY: int = 80
    COLOR: tuple[float, float, float, float] = (0, 1, 1, 1)

    NUM: int = 1
    INITIAL_POSITION: list[Position] = []


@jax_dataclass
class Nest:
    POSITION: Position = Position(0.0, 0.0)
    RADIUS: float = 1.0
    HEIGHT: float = 0.01
    COLOR: tuple[float, float, float, float] = (0, 1, 0, 1)


@jax_dataclass
class Loss:
    OFFSET_NEST_AND_FOOD: float = 0
    SIGMA_NEST_AND_FOOD: jnp.ndarray = calc_loss_sigma(4, 0.01)
    GAIN_NEST_AND_FOOD: int = 1

    OFFSET_ROBOT_AND_FOOD: float = Robot.RADIUS + Food.RADIUS
    SIGMA_ROBOT_AND_FOOD: jnp.ndarray = calc_loss_sigma(1, 0.3)
    GAIN_ROBOT_AND_FOOD: float = 0.01

    REGULARIZATION_COEFFICIENT: int = 0


@jax_dataclass
class Simulation:
    TIME_STEP: float = 0.01
    TIME_LENGTH: int = 60  # Unit is Seconds

    WORLD_WIDTH: float = 10.0
    WORLD_HEIGHT: float = 10.0

    WALL_THICKNESS: float = 1
    WALL_HEIGHT: float = 1

    TEMPERATURE: float = 300.0  # Kelvin


@jax_dataclass
class Storage:
    SAVE_INDIVIDUALS: bool = True
    SAVE_DIRECTORY: str = "./results"
    SAVE_INTERVAL: int = 10  # Save every N generations
    TOP_N: int = 0  # Save top N individuals, 0 means save all
    ASSET_DIRECTORY: str = "./assets"


@jax_dataclass
class Device:
    ENABLE_CUDA: bool = False


@jax_dataclass
class Pheromone:
    ACTIVE: bool = False
    CELL_SIZE: float = 0.1
    WIDTH_NUM: int = int(Simulation.WORLD_WIDTH / CELL_SIZE)
    HEIGHT_NUM: int = int(Simulation.WORLD_HEIGHT / CELL_SIZE)
    ITERATIONS_PER_STEP: int = 1
    EVAPORATION_RATE: float = 0.1
    DECREASE_RATE: float = 0.0
    MATERIAL: Any = ETHANOL


@jax_dataclass
class Settings:
    """
    Basically, the attributes' unit is meter.
    JAX-compatible configuration dataclass.
    """

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


def settings_to_dict(settings: Settings) -> dict[str, Any]:
    """JAX-compatible function to convert settings to dictionary."""
    import jax.tree_util as jtu
    return jtu.tree_map(lambda x: x, settings)


def compare_settings(settings1: Settings, settings2: Settings) -> dict[str, list[tuple[str, Any, ...] | tuple[str, Any]]]:
    """JAX-compatible function to compare two settings objects."""
    import jax.tree_util as jtu
    
    def are_equal(x: Any, y: Any) -> bool:
        try:
            return jnp.array_equal(x, y) if hasattr(x, 'shape') else x == y
        except (TypeError, ValueError, AttributeError):
            return str(x) == str(y)
    
    differences: list[tuple[str, Any, Any]] = []
    identical: list[tuple[str, Any]] = []
    
    leaves1, tree_def1 = jtu.tree_flatten(settings1)
    leaves2, tree_def2 = jtu.tree_flatten(settings2)
    
    if tree_def1 != tree_def2:
        return {"difference": [("structure", "different_structure", "different_structure")], "identical": []}
    
    for i, (val1, val2) in enumerate(zip(leaves1, leaves2)):
        if are_equal(val1, val2):
            identical.append((f"leaf_{i}", val1))
        else:
            differences.append((f"leaf_{i}", val1, val2))
    
    return {"difference": differences, "identical": identical}


