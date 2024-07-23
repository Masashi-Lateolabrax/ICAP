# from lib.utils import BodyObject, IntervalTimer
#
# from .brain import NeuralNetwork
# from .... import Settings
#
#
# # def update(self, y: np.ndarray):
# #     self.movement = (y[0] + y[1]) * 0.5 * Settings.Structure.Robot.MOVE_SPEED
# #     self.rotation = (y[0] - y[1]) * 0.5 * Settings.Structure.Robot.TURN_SPEED
#
#
# class RobotLiBuf:
#     def __init__(self, sensor_buf: DistanceMeasureBuf):
#         self.sensor_buf = sensor_buf
#         self.calc_buf_for_direction = np.zeros(3)
#
#
# class RobotLI(BodyObject):
#     def __init__(
#             self,
#             model: mujoco.MjModel,
#             data: mujoco.MjData,
#             bot_id: int,
#             brain: NeuralNetwork,
#             buf: RobotLiBuf | None = None
#     ):
#         super().__init__(model, data, Settings.Structure.Robot.NAMES(bot_id)["body"])
#
#         self.buf = RobotLiBufif
#         buf is None
#
#         self.cam_name = GeneralSetting.Robot.NAMES(bot_id)["camera"]
#
#         self._think_interval = IntervalTimer(GeneralSetting.Robot.THINK_INTERVAL / GeneralSetting.Simulation.TIMESTEP)
#
#         self._sensor = DistanceMeasure(Settings.Structure.Robot.Sensor().)
#         self._actuator = _Actuator(data, bot_id)
#         self.brain = brain
#
#         self._output = np.zeros(3)
#         self.debug_data: dict[str, np.ndarray] | None = None
#         self._direction_buf = calc_buf_for_direction
#
#         self.buf = RobotLiBuf(self._sensor) if buf is None else buf
#
#     def exec(self, model: mujoco.MjModel, data: mujoco.MjData, act=True, debug=False):
#         mujoco.mju_rotVecQuat(self._direction_buf, [0, 1, 0], self.get_body().xquat)
#
#         if self._think_interval.count():
#             self._output[:] = self.brain.forward(from_sensor, from_pheromone).detach().numpy()
#
#         if self._state.do_think():
#             y =
#             self._actuator.update(y)
#
#             if debug:
#                 self.debug_data = self.brain.debugger.get_buf()
#
#         if act:
#             self._actuator.act(bot_direction)
#
#         return self._pheromone

import math
import collections
from collections.abc import Sequence

import mujoco
import numpy as np
import torch
import torch.nn as nn

from lib.sensor import TrigonoOmniSensor, TrigonoOmniSensorBuf, DistanceMeasure2, DistanceMeasure2Buf

from ....settings import Settings
from .. import _robot as interfaces
from ..._world import World

from .brain import Brain


class _Buffer(interfaces.RobotBuf):
    def __init__(self, world: World):
        super().__init__()

        self.tri_sensor_buf_for_robot = TrigonoOmniSensorBuf()
        self.tri_sensor_buf_for_safezone = TrigonoOmniSensorBuf()
        self.dist_sensor_buf = DistanceMeasure2Buf(4)
        self.input_buf = torch.zeros(5)

        self.safezone_pos = np.array(world.safezone_pos)


class RobotFactory(interfaces.RobotFactory):
    def __init__(self, brain_sharing: bool, buffer_sharing: bool):
        self.brain = None if brain_sharing else Brain()
        self.buffer_sharing = buffer_sharing
        self.buf: _Buffer | None = None

    def get_dim(self):
        return Brain().num_dim()

    def manufacture(self, world: World, bot_id: int, brain_para: Sequence[float]):
        brain = Brain() if self.brain is None else self.brain
        brain.load_para(brain_para)

        if not self.buffer_sharing:
            buf = _Buffer(world)
        elif self.buf is None:
            buf = self.buf = _Buffer(world)
        else:
            buf = self.buf

        return RobotTR(world, bot_id, brain, buf)


class RobotTR(interfaces.Robot):
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

        self.bot_id = bot_id
        self._brain = brain
        self._buf = buf

        self._tri_sensor_for_bot = TrigonoOmniSensor(
            Settings.Characteristics.Sensor.Trigonometric.GAIN_FOR_BOT,
            self._buf.tri_sensor_buf_for_robot
        )
        self._tri_sensor_for_safezone = TrigonoOmniSensor(
            Settings.Characteristics.Sensor.Trigonometric.GAIN_FOR_SAFEZONE,
            self._buf.tri_sensor_buf_for_safezone
        )
        self._dist_sensor = DistanceMeasure2(4, self._buf.dist_sensor_buf)

    def preparation(self, world: World):
        mujoco.mju_rotVecQuat(self._buf.bot_direction, [0, 1, 0], self.get_body().xquat)
        pos = self.get_body().xpos[0:2]

        other_bot_index_mask = [self.bot_id != i for i in range(Settings.Task.NUM_ROBOT)]
        other_bot_positions = world.bot_pos[other_bot_index_mask, :]

        self._buf.input_buf[0], self._buf.input_buf[1] = sensed_bot = self._tri_sensor_for_bot.measure(
            pos, self._buf.bot_direction[0:2, 0], other_bot_positions
        )
        self._buf.input_buf[2], self._buf.input_buf[3] = sensed_safezone = self._tri_sensor_for_safezone.measure(
            pos, self._buf.bot_direction[0:2, 0], self._buf.safezone_pos
        )

        pheromone = world.pheromone.get_gas(self) / Settings.Characteristics.Pheromone.SATURATION_VAPOR
        self._buf.input_buf[4] = pheromone

        if isinstance(self._input_buf, interfaces.RobotDebugBuf):
            if self._input_buf.input.shape != (1, 5, 3):
                self._input_buf.input.resize((1, 5, 3))
            self._input_buf.input[0, 0:2, :] = self._tri_sensor_for_bot.convert_to_img(sensed_bot)
            self._input_buf.input[0, 2:4, :] = self._tri_sensor_for_safezone.convert_to_img(sensed_safezone)
            self._input_buf.input[0, 4, :] = pheromone * 255

    def update_state(self, world: World):
        action = self._brain.forward(self._buf.input_buf).detach().numpy()
        self.movement_power = (action[0] + action[1]) * 0.5 * Settings.Environment.Robot.MOVE_SPEED
        self.rotation_power = (action[0] - action[1]) * 0.5 * Settings.Environment.Robot.TURN_SPEED
        self.pheromone_secretion = action[2] * Settings.Environment.Robot.MAX_SECRETION * Settings.Simulation.TIMESTEP
