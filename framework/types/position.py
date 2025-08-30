import jax
from flax.struct import dataclass as jax_dataclass


@jax_dataclass
class Position:
    x: float
    y: float

    def as_array(self) -> jax.Array:
        return jax.numpy.array([self.x, self.y])


@jax_dataclass
class Position3d:
    x: float
    y: float
    z: float

    def to_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@jax_dataclass
class RobotLocation:
    x: float
    y: float
    angle: float

    @property
    def position(self) -> Position:
        return Position(self.x, self.y)

    @property
    def theta(self) -> float:
        return self.angle
