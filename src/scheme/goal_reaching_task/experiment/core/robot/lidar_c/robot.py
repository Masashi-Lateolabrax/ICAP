import math
import collections
from collections.abc import Sequence

import mujoco
import torch
import torch.nn as nn

from lib.sensor import DistanceMeasure2, DistanceMeasure2Buf

from ....settings import Settings
from .. import _robot as interfaces
from ..._world import World

from .brain import Brain


def _normal_distribution(x) -> float:
    mean = Settings.Environment.Robot.Sensor.Lidar.DownScale.MEAN
    sigma = Settings.Environment.Robot.Sensor.Lidar.DownScale.SIGMA
    variance_2 = 2 * math.pow(sigma, 2)
    return math.exp(-math.pow(x - mean, 2) / variance_2) / math.sqrt(math.pi * variance_2)


class _Buffer(interfaces.RobotBuf):
    def __init__(self):
        super().__init__()

        self.dist_measure_buf = DistanceMeasureBuf(Settings.Characteristics.Sensor.Lidar.NUM_LASER)
        self.sight = torch.zeros(Settings.Environment.Robot.Sensor.Lidar.DownScale.DIM_OUTPUT)
        self.pheromone = torch.zeros(1)


class _DownScaleModel(nn.Module):
    def __init__(self):
        super(_DownScaleModel, self).__init__()

        self.requires_grad_(False)

        kernel_size = Settings.Environment.Robot.Sensor.Lidar.KERNEL_SIZE
        self.sequence = torch.nn.Sequential(collections.OrderedDict([
            ("padding", torch.nn.CircularPad1d(int(kernel_size * 0.5 + 0.5))),
            ("convolve", torch.nn.Conv1d(1, 1, kernel_size, int(kernel_size * 0.5), bias=False)),
        ]))
        self.sequence.requires_grad_(False)
        self.sequence.convolve.weight.copy_(torch.tensor(
            [_normal_distribution(x) for x in range(0, kernel_size)],
            dtype=torch.float32, requires_grad=False
        ))

    def forward(self, x):
        return self.sequence.forward(x)


class RobotFactory(interfaces.RobotFactory):
    def __init__(self, brain_sharing: bool, buffer_sharing: bool):
        self.brain = None if brain_sharing else Brain()
        self.buf = None if buffer_sharing else _Buffer()

    def get_dim(self):
        return Brain().num_dim()

    def manufacture(self, world: World, bot_id: int, brain_para: Sequence[float]):
        brain = Brain() if self.brain is None else self.brain
        buf = _Buffer() if self.buf is None else self.buf
        brain.load_para(brain_para)
        return RobotC(world, bot_id, brain, buf)


class RobotC(interfaces.Robot):
    @staticmethod
    def get_dim():
        return Brain().num_dim()

    def __init__(
            self,
            world: World,
            bot_id: int,
            brain: Brain,
            buf: _Buffer,
    ):
        super().__init__(world, bot_id, buf)

        self._brain = brain
        self._buf = buf

        self._sensor = DistanceMeasure(
            Settings.Characteristics.Sensor.Lidar.NUM_LASER,
            buf.dist_measure_buf
        )

        self._down_scale = _DownScaleModel()

    def preparation(self, world: World):
        mujoco.mju_rotVecQuat(self._buf.bot_direction, [0, 1, 0], self.get_body().xquat)

        img = self._sensor.measure_with_brightness_img(
            m=world.model,
            d=world.data,
            position=self.body.xpos,
            direction=self._buf.bot_direction,
            color_map=Settings.Environment.World.COLOR_MAP,
            gain=Settings.Characteristics.Sensor.Lidar.GAIN,
            bodyexclude=self.body_id,
            cutoff=Settings.Characteristics.Sensor.Lidar.CUTOFF
        )
        self._buf.sight = self._down_scale.forward(torch.from_numpy(img))

        self._buf.pheromone[0] = pheromone = world.pheromone.get_gas(self)

        if isinstance(self._input_buf, interfaces.RobotDebugBuf):
            if self._input_buf.input.shape != (1, Settings.Environment.Robot.Sensor.Lidar.DownScale.DIM_OUTPUT + 1, 3):
                self._input_buf.input.resize((1, Settings.Environment.Robot.Sensor.Lidar.DownScale.DIM_OUTPUT + 1, 3))
            detached = self._buf.sight.detach().numpy()
            self._input_buf.input[:, 0:-1, 0] = detached
            self._input_buf.input[:, 0:-1, 1] = detached
            self._input_buf.input[:, 0:-1, 2] = detached
            self._input_buf.input[:, -1, :] = pheromone

    def update_state(self, world: World):
        action = self._brain.forward(self._buf.sight, self._buf.pheromone).detach().numpy()
        self.movement_power = (action[0] + action[1]) * 0.5 * Settings.Environment.Robot.MOVE_SPEED
        self.rotation_power = (action[0] - action[1]) * 0.5 * Settings.Environment.Robot.TURN_SPEED
        self.pheromone_secretion = action[2] * Settings.Environment.Robot.MAX_SECRETION
