import abc
from collections.abc import Sequence
import enum

import numpy as np
import mujoco
import torch

from libs.mujoco_utils.objects import BodyObject

from ....prerude import Settings
from ....utils import robot_names
from .._environment import Environment

from ._pheromone_tank import PheromoneTank


class RobotAction(enum.Enum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    SECRETION = 4
    STOP = 5


class RobotDebugBuf:
    def __init__(self):
        self.input = np.zeros((1, 1), dtype=np.uint8)


class RobotBuf:
    def __init__(self):
        self.bot_direction = np.zeros((3, 1))
        self.velocity = torch.zeros(2)
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

        self._sen_vel = d.sensor(name_table["v_sen"])

        self._buf = buf
        self._input_buf: RobotDebugBuf | None = None

    def take_action(self, act: RobotAction, env: Environment):
        mv = self._buf.bot_direction * Settings.Characteristic.Robot.MOVE_SPEED
        match act:
            case RobotAction.MOVE_FORWARD:
                self._act_move_x.ctrl[0] = mv[0]
                self._act_move_y.ctrl[0] = mv[1]
            case RobotAction.MOVE_BACKWARD:
                self._act_move_x.ctrl[0] = -mv[0]
                self._act_move_y.ctrl[0] = -mv[1]
            case RobotAction.TURN_LEFT:
                self._act_rot.ctrl[0] += Settings.Characteristic.Robot.TURN_SPEED
            case RobotAction.TURN_RIGHT:
                self._act_rot.ctrl[0] -= Settings.Characteristic.Robot.TURN_SPEED
            case RobotAction.SECRETION:
                secretion = self.pheromone_tank.secretion(Settings.Characteristic.Robot.SECRETION)
                env.pheromone.add_liquid(self, secretion)

    @abc.abstractmethod
    def think(self, env: Environment) -> RobotAction:
        raise NotImplemented

    def act(self, env: Environment):
        mujoco.mju_rotVecQuat(self._buf.bot_direction, [0, 1, 0], self.get_body().xquat)
        self._buf.velocity.copy_(torch.from_numpy(self._sen_vel.data[0:2]))

        # dn = np.linalg.norm(env.nest_pos - self.get_body().xpos[0:2])
        # if dn < Settings.Task.Nest.SIZE:
        #     self.pheromone_tank.fill()
        # self._buf.tank[0] = self.pheromone_tank.remain()
        decision = self.think(env)

        self.take_action(decision, env)

    def get_direction(self):
        return self._buf.bot_direction[0:2, 0]

    def set_input_buf(self, buf: RobotDebugBuf | None):
        self._input_buf = buf
