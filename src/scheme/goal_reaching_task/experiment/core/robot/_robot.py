import abc
from collections.abc import Sequence

import numpy as np

from lib.mujoco_utils.mujoco_obj import BodyObject

from ...settings import Settings
from .._world import World


class RobotDebugBuf:
    def __init__(self):
        self.input = np.zeros((1, 1), dtype=np.uint8)


class RobotBuf:
    def __init__(self):
        self.bot_direction = np.zeros((3, 1))


class RobotFactory(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def manufacture(self, world: World, bot_id: int, brain_para: Sequence[float]):
        raise NotImplemented

    @abc.abstractmethod
    def get_dim(self) -> int:
        raise NotImplemented


class Robot(BodyObject, metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def get_dim():
        raise NotImplemented

    def __init__(self, world: World, bot_id: int, shared_buf: RobotBuf):
        name_table = Settings.Environment.Robot.NAMES(bot_id)

        super().__init__(world.model, world.data, name_table["body"])

        self.cam_name = name_table["camera"]

        self._act_rot = world.data.actuator(Settings.Environment.Robot.NAMES(bot_id)["r_act"])
        self._act_move_x = world.data.actuator(Settings.Environment.Robot.NAMES(bot_id)["x_act"])
        self._act_move_y = world.data.actuator(Settings.Environment.Robot.NAMES(bot_id)["y_act"])
        self.movement_power = 0
        self.rotation_power = 0

        self.pheromone_secretion = 0

        self._buf = shared_buf
        self._input_buf: RobotDebugBuf | None = None

    @abc.abstractmethod
    def preparation(self, world: World):
        raise NotImplemented

    @abc.abstractmethod
    def update_state(self, world: World):
        raise NotImplemented

    def move(self, world: World):
        move_vec = self._buf.bot_direction * self.movement_power
        self._act_rot.ctrl[0] = self.rotation_power
        self._act_move_x.ctrl[0] = move_vec[0, 0]
        self._act_move_y.ctrl[0] = move_vec[1, 0]
        world.pheromone.add_liquid(self, self.pheromone_secretion)

    def act(self, world: World, update_power: bool):
        self.preparation(world)
        if update_power:
            self.update_state(world)
        self.move(world)

    def get_direction(self):
        return self._buf.bot_direction

    def set_input_buf(self, buf: RobotDebugBuf | None):
        self._input_buf = buf
