from collections.abc import Sequence
import math

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

        self.sight = torch.zeros(4)
        self.nest = torch.zeros(2)
        self.pheromone = torch.zeros(1)


class RobotFactory(interfaces.RobotFactory):
    def __init__(self, brain_sharing: bool, buffer_sharing: bool, debug: bool):
        self.debug = debug
        self.brain = None if brain_sharing else Brain(self.debug)
        self.buffer_sharing = buffer_sharing
        self.buf: _Buffer | None = None

    def get_dim(self):
        return Brain().num_dim() + 1

    def manufacture(self, env: Environment, bot_id: int, para: Sequence[float]):
        brain = Brain(self.debug) if self.brain is None else self.brain
        brain.load_para(para[0:-1])
        secretion = 0.5 * (math.tanh(para[-1]) + 1) * Settings.Characteristic.Robot.MAX_SECRETION

        if not self.buffer_sharing:
            buf = _Buffer()
        elif self.buf is None:
            buf = self.buf = _Buffer()
        else:
            buf = self.buf

        return RobotTR(env, bot_id, brain, secretion, buf)


class RobotTR(interfaces.Robot):
    @staticmethod
    def get_dim():
        return Brain().num_dim() + 1

    def __init__(
            self,
            env: Environment,
            bot_id: int,
            brain: Brain,
            secretion: float,
            buf: _Buffer,
    ):
        super().__init__(env, bot_id, buf)

        self._timer = IntervalTimer(
            Settings.Characteristic.Robot.THINKING_INTERVAL
        )

        self.bot_id = bot_id
        self._brain = brain
        self._secretion = secretion
        self._buf = buf

        self._tri_sensor_for_bot = TrigonoOmniSensor(
            lambda dist: np.reciprocal(Settings.Characteristic.Robot.SENSOR_PRECISION[0] * dist + 1),
            self._buf.tri_sensor_buf_for_robot
        )
        self._tri_sensor_for_food = TrigonoOmniSensor(
            lambda dist: np.reciprocal(Settings.Characteristic.Robot.SENSOR_PRECISION[1] * dist + 1),
            self._buf.tri_sensor_buf_for_food
        )

    def _update_input(self, env: Environment):
        direction = self.get_direction()
        pos = env.bot_pos[self.bot_id]

        other_bot_index_mask = [self.bot_id != i for i in range(len(Settings.Task.Robot.POSITIONS))]
        other_bot_positions = env.bot_pos[other_bot_index_mask, :]

        self._buf.sight[0], self._buf.sight[1] = self._tri_sensor_for_bot.measure(
            pos, direction, other_bot_positions
        )
        self._buf.sight[2], self._buf.sight[3] = self._tri_sensor_for_food.measure(
            pos, direction, env.food_pos
        )

        self._buf.nest[0], self._buf.nest[1] = direction_sensor(
            pos, direction, env.nest_pos, Settings.Task.Nest.SIZE
        )

        self._buf.pheromone[0] = env.pheromone.get_gas(self) / env.pheromone.get_sv()

    def update(self, env: Environment):
        if self._timer.count(Settings.Simulation.TIMESTEP):
            self._update_input(env)

            act = self._brain.forward(
                self._buf.sight,
                self._buf.nest,
                self._buf.pheromone,
                self._buf.velocity,
                # self._buf.tank
            ).detach().numpy()

            m_power = (act[0] + act[1]) * 0.5 * Settings.Characteristic.Robot.MOVE_SPEED
            r_power = (act[0] - act[1]) * 0.5 * Settings.Characteristic.Robot.TURN_SPEED
            secretion = act[2] * self._secretion * Settings.Simulation.TIMESTEP

            self.set_powers(m_power, r_power, secretion)
