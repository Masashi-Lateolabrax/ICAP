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


@dataclasses.dataclass
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