import abc
from collections.abc import Sequence

import numpy as np
import mujoco
import torch

from libs.mujoco_utils.objects import BodyObject

from ....settings import Settings
from ....utils import robot_names
from .._environment import Environment

from ._pheromone_tank import PheromoneTank


class RobotDebugBuf:
    def __init__(self):
        self.input = np.zeros((1, 1), dtype=np.uint8)


class RobotBuf:
    def __init__(self):
        self.bot_direction = np.zeros((3, 1))
        self.tank = torch.zeros(1)


class RobotFactory(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def manufacture(self, world: Environment, bot_id: int, para: Sequence[float]):
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
        m = env.get_model()
        d = env.get_data()

        super().__init__(m, d, name_table["body"])

        self.pheromone_tank = PheromoneTank(
            Settings.Characteristic.Robot.TANK_SIZE
        )

        self.cam_name = name_table["camera"]

        self._act_rot = d.actuator(name_table["r_act"])
        self._act_move_x = d.actuator(name_table["x_act"])
        self._act_move_y = d.actuator(name_table["y_act"])

        self._movement_power = 0
        self._rotation_power = 0
        self._pheromone_secretion = 0

        self._buf = buf
        self._input_buf: RobotDebugBuf | None = None

    def set_powers(self, movement_power, rotation_power, pheromone_secretion):
        self._movement_power = movement_power
        self._rotation_power = rotation_power
        self._pheromone_secretion = pheromone_secretion

    @abc.abstractmethod
    def update(self, env: Environment):
        raise NotImplemented

    def act(self, env: Environment):
        mujoco.mju_rotVecQuat(self._buf.bot_direction, [0, 1, 0], self.get_body().xquat)

        dn = np.linalg.norm(env.nest_pos - self.get_body().xpos[0:2])
        if dn < Settings.Task.Nest.SIZE:
            self.pheromone_tank.fill()
        self._buf.tank[0] = self.pheromone_tank.remain()

        self.update(env)

        move_vec = self._buf.bot_direction * self._movement_power
        self._act_rot.ctrl[0] = self._rotation_power
        self._act_move_x.ctrl[0] = move_vec[0, 0]
        self._act_move_y.ctrl[0] = move_vec[1, 0]

        secretion = self.pheromone_tank.secretion(self._pheromone_secretion)
        env.pheromone.add_liquid(self, secretion)

    def get_direction(self):
        return self._buf.bot_direction[0:2, 0]

    def set_input_buf(self, buf: RobotDebugBuf | None):
        self._input_buf = buf
