import dataclasses

import mujoco


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


class RobotLocation:
    def __init__(self, x: float, y: float, theta: float):
        self.position = Position(x, y)
        self.theta = theta  # Unit is radian

    @property
    def x(self) -> float:
        return self.position.x

    @property
    def y(self) -> float:
        return self.position.y


class RobotSpec:
    def __init__(
            self,
            center_site: mujoco._specs.MjsSite,
            front_site: mujoco._specs.MjsSite,
            free_join: mujoco._specs.MjsJoint,
            x_act: mujoco._specs.MjsActuator,
            y_act: mujoco._specs.MjsActuator,
            r_act: mujoco._specs.MjsActuator
    ):
        self.center_site = center_site
        self.front_site = front_site
        self.free_join = free_join
        self.x_act = x_act
        self.y_act = y_act
        self.r_act = r_act


class RobotValues:
    def __init__(self, data: mujoco.MjData, spec: RobotSpec):
        self.center_site: mujoco._structs._MjDataSiteViews = data.site(spec.center_site.name)
        self.front_site: mujoco._structs._MjDataSiteViews = data.site(spec.front_site.name)
        self.free_join: mujoco._structs._MjDataJointViews = data.joint(spec.free_join.name)
        self.x_act: mujoco._structs._MjDataActuatorViews = data.actuator(spec.x_act.name)
        self.y_act: mujoco._structs._MjDataActuatorViews = data.actuator(spec.y_act.name)
        self.r_act: mujoco._structs._MjDataActuatorViews = data.actuator(spec.r_act.name)
