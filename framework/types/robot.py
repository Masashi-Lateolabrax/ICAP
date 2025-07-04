import mujoco
import numpy as np

from .position import Position


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
            z_act: mujoco._specs.MjsActuator,
            r_act: mujoco._specs.MjsActuator
    ):
        self.center_site = center_site
        self.front_site = front_site
        self.free_join = free_join
        self.x_act = x_act
        self.y_act = y_act
        self.z_act = z_act
        self.r_act = r_act


class RobotValues:
    _d = 1
    _v = 1
    _z = 1
    _matrix = np.zeros((2, 2))

    @staticmethod
    def _update_matrix():
        RobotValues._matrix = np.array([
            [RobotValues._v * 0.5, RobotValues._v * 0.5],
            [- RobotValues._v / RobotValues._d, RobotValues._v / RobotValues._d]
        ])

    @staticmethod
    def set_distance_between_wheels(d: float):
        RobotValues._d = d
        RobotValues._update_matrix()

    @staticmethod
    def set_max_speed(v: float):
        RobotValues._v = v
        RobotValues._update_matrix()

    @staticmethod
    def set_robot_height(h: float):
        RobotValues._z = h * 0.5 + 0.01

    def __init__(self, data: mujoco.MjData, spec: RobotSpec):
        self._center_site: mujoco._structs._MjDataSiteViews = data.site(spec.center_site.name)
        self._front_site: mujoco._structs._MjDataSiteViews = data.site(spec.front_site.name)
        self._free_join: mujoco._structs._MjDataJointViews = data.joint(spec.free_join.name)
        self._x_act: mujoco._structs._MjDataActuatorViews = data.actuator(spec.x_act.name)
        self._y_act: mujoco._structs._MjDataActuatorViews = data.actuator(spec.y_act.name)
        self._z_act: mujoco._structs._MjDataActuatorViews = data.actuator(spec.z_act.name)
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
        return self._xdirection()[0:2]

    @property
    def direction(self):
        mujoco.mju_mulMatVec3(self._direction_buf, self._center_site.xmat, self._xdirection())
        self._direction_buf[2] = 0
        self._direction_buf /= np.linalg.norm(self._direction_buf)
        return self._direction_buf[0:2]

    def act(self, right_wheel, left_wheel):
        power_and_torque = np.dot(self._matrix, [right_wheel, left_wheel])

        power = self.xdirection * power_and_torque[0]
        self._x_act.ctrl[0] = power[0]
        self._y_act.ctrl[0] = power[1]
        self._z_act.ctrl[0] = RobotValues._z
        self._r_act.ctrl[0] = power_and_torque[1]