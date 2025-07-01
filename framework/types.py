import dataclasses

import mujoco
import numpy as np


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
        self._center_site: mujoco._structs._MjDataSiteViews = data.site(spec.center_site.name)
        self._front_site: mujoco._structs._MjDataSiteViews = data.site(spec.front_site.name)
        self._free_join: mujoco._structs._MjDataJointViews = data.joint(spec.free_join.name)
        self._x_act: mujoco._structs._MjDataActuatorViews = data.actuator(spec.x_act.name)
        self._y_act: mujoco._structs._MjDataActuatorViews = data.actuator(spec.y_act.name)
        self._r_act: mujoco._structs._MjDataActuatorViews = data.actuator(spec.r_act.name)

        self._xdirection_buf = np.zeros(3)
        self._direction_buf = np.zeros(3)

    @property
    def site(self):
        return self._center_site

    @property
    def xpos(self):
        return self._center_site.xpos[0:2]

    def _xdirection(self):
        self._xdirection_buf[:] = self._front_site.xpos[0:3]
        self._xdirection_buf -= self._center_site.xpos[0:3]
        self._xdirection_buf[2] = 0
        self._xdirection_buf /= np.linalg.norm(self._xdirection_buf)
        return self._xdirection_buf

    @property
    def xdirection(self):
        return self._xdirection_buf[0:2]

    @property
    def direction(self):
        mujoco.mju_mulMatVec(self._direction_buf, self._center_site.xmat, self._xdirection())
        self._direction_buf[2] = 0
        self._direction_buf /= np.linalg.norm(self._direction_buf)
        return self._direction_buf[0:2]

    def act(self, move, rot):
        power = self.direction * move
        self._x_act.ctrl[0] = power[0]
        self._y_act.ctrl[0] = power[1]
        self._r_act.ctrl[0] = rot


class FoodSpec:
    def __init__(
            self,
            center_site: mujoco._specs.MjsSite,
            free_join: mujoco._specs.MjsJoint,
            velocimeter: mujoco._specs.MjsSensor
    ):
        self.center_site = center_site
        self.free_join = free_join
        self.velocimeter = velocimeter


class FoodValues:
    def __init__(self, data: mujoco.MjData, spec: FoodSpec):
        self._center_site = data.site(spec.center_site.name)
        self._free_join = data.joint(spec.free_join.name)
        self._velocimeter = data.sensor(spec.velocimeter.name)

    @property
    def site(self):
        return self._center_site

    @property
    def xpos(self):
        return self._center_site.xpos[0:2]
