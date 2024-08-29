import abc
from collections.abc import Sequence

import numpy as np
import mujoco

from libs.mujoco_utils.objects import BodyObject

from ....utils import robot_names
from .._environment import Environment


class RobotDebugBuf:
    def __init__(self):
        self.input = np.zeros((1, 1), dtype=np.uint8)


class RobotBuf:
    def __init__(self):
        self.bot_direction = np.zeros((3, 1))


class RobotFactory(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def manufacture(self, world: Environment, bot_id: int, brain_para: Sequence[float]):
        raise NotImplemented

    @abc.abstractmethod
    def get_dim(self) -> int:
        raise NotImplemented


class Robot(BodyObject, metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def get_dim():
        raise NotImplemented

    def __init__(self, env: Environment, bot_id: int, buf: RobotBuf):
        name_table = robot_names(bot_id)

        super().__init__(env.m, env.d, name_table["body"])

        self.cam_name = name_table["camera"]

        self._act_rot = env.d.actuator(name_table["r_act"])
        self._act_move_x = env.d.actuator(name_table["x_act"])
        self._act_move_y = env.d.actuator(name_table["y_act"])
        self.movement_power = 0
        self.rotation_power = 0

        self._buf = buf
        self._input_buf: RobotDebugBuf | None = None

    @abc.abstractmethod
    def update(self, env: Environment):
        raise NotImplemented

    def move(self):
        move_vec = self._buf.bot_direction * self.movement_power
        self._act_rot.ctrl[0] = self.rotation_power
        self._act_move_x.ctrl[0] = move_vec[0, 0]
        self._act_move_y.ctrl[0] = move_vec[1, 0]

    def act(self, env: Environment):
        self.update(env)
        self.move()

    def get_direction(self):
        mujoco.mju_rotVecQuat(self._buf.bot_direction, [0, 1, 0], self.get_body().xquat)
        return self._buf.bot_direction[0:2, 0]

    def set_input_buf(self, buf: RobotDebugBuf | None):
        self._input_buf = buf
