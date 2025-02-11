import numpy as np
import mujoco

from mujoco._structs import _MjDataBodyViews, _MjDataJointViews

from scheme.pushing_food_with_pheromone.lib.world import WorldClock
from scheme.pushing_food_with_pheromone.lib.utilities import Cache


class RobotData(Cache):
    def __init__(self, body: _MjDataBodyViews, joint_r: _MjDataJointViews, timer: WorldClock):
        super().__init__(timer)

        self.pos = body.xpos
        self.angle = joint_r.qpos

        self._quat: np.ndarray = np.zeros(4)
        self._direction: np.ndarray = np.zeros(3)

    def _update_direction(self):
        mujoco.mju_axisAngle2Quat(self._quat, [0, 0, 1], self.angle)
        mujoco.mju_rotVecQuat(self._direction, [0, 1, 0], self._quat)

    def _update(self):
        self._update_direction()

    @property
    def direction(self):
        self.update()
        return self._direction
