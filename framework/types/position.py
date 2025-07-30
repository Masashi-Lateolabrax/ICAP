import dataclasses


@dataclasses.dataclass
class Position:
    x: float
    y: float


@dataclasses.dataclass
class Position3d:
    x: float
    y: float
    z: float

    def to_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z