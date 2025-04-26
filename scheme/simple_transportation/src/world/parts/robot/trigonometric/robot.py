from collections.abc import Sequence

import numpy as np
import torch

from libs.sensor import TrigonoOmniSensor, TrigonoOmniSensorBuf, direction_sensor
from libs.utils import IntervalTimer

from .....settings import Settings
from ... import Environment
from .. import _robot as interfaces
from .brain import Brain


class _Buffer(interfaces.RobotBuf):
    def __init__(self):
        super().__init__()

        self.tri_sensor_buf_for_robot = TrigonoOmniSensorBuf()
        self.tri_sensor_buf_for_food = TrigonoOmniSensorBuf()

        self.input_buf = torch.zeros(6)


class RobotFactory(interfaces.RobotFactory):
    def __init__(self, brain_sharing: bool, buffer_sharing: bool):
        self.brain = None if brain_sharing else Brain()
        self.buffer_sharing = buffer_sharing
        self.buf: _Buffer | None = None

    def get_dim(self):
        return Brain().num_dim()

    def manufacture(self, env: Environment, bot_id: int, brain_para: Sequence[float]):
        brain = Brain() if self.brain is None else self.brain
        brain.load_para(brain_para)

        if not self.buffer_sharing:
            buf = _Buffer()
        elif self.buf is None:
            buf = self.buf = _Buffer()
        else:
            buf = self.buf

        return RobotTR(env, bot_id, brain, buf)


class RobotTR(interfaces.Robot):
    @staticmethod
    def get_dim():
        return Brain().num_dim()

    def __init__(
            self,
            env: Environment,
            bot_id: int,
            brain: Brain,
            buf: _Buffer,
    ):
        super().__init__(env, bot_id, buf)

        self.bot_id = bot_id
        self._brain = brain
        self._buf = buf

        self._tri_sensor_for_bot = TrigonoOmniSensor(
            lambda dist: np.reciprocal(Settings.Structure.Robot.SENSOR_PRECISION[0] * dist + 1),
            self._buf.tri_sensor_buf_for_robot
        )
        self._tri_sensor_for_food = TrigonoOmniSensor(
            lambda dist: np.reciprocal(Settings.Structure.Robot.SENSOR_PRECISION[1] * dist + 1),
            self._buf.tri_sensor_buf_for_food
        )

    def _update_input(self, env: Environment):
        input_ = self._buf.input_buf

        direction = self.get_direction()
        pos = env.bot_pos[self.bot_id]

        other_bot_index_mask = [self.bot_id != i for i in range(len(Settings.Task.Robot.POSITIONS))]
        other_bot_positions = env.bot_pos[other_bot_index_mask, :]

        self._buf.input_buf[0], self._buf.input_buf[1] = self._tri_sensor_for_bot.measure(
            pos, direction, other_bot_positions
        )
        self._buf.input_buf[2], self._buf.input_buf[3] = self._tri_sensor_for_food.measure(
            pos, direction, env.food_pos
        )
        self._buf.input_buf[4], self._buf.input_buf[5] = direction_sensor(
            pos, direction, env.nest_pos, Settings.Task.Nest.SIZE
        )

        return input_

    def _update_power(self, input_):
        action = self._brain.forward(input_).detach().numpy()
        self.movement_power = (action[0] + action[1]) * 0.5 * Settings.Structure.Robot.MOVE_SPEED
        self.rotation_power = (action[0] - action[1]) * 0.5 * Settings.Structure.Robot.TURN_SPEED
        self.pheromone_secretion = action[2] * Settings.Structure.Robot.MAX_SECRETION * Settings.Simulation.TIMESTEP

    def update(self, env: Environment):
        input_ = self._update_input(env)
        self._update_power(input_)
