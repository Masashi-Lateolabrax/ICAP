import numpy as np
import mujoco

from mujoco._structs import _MjDataBodyViews, _MjDataJointViews

from scheme.transportation_with_pheromone.lib.world import WorldClock
from scheme.transportation_with_pheromone.lib.utilities import Cache


class RobotProperty(Cache):
    def __init__(self, body: _MjDataBodyViews, size: float, joint_r: _MjDataJointViews, timer: WorldClock):
        super().__init__(timer)

        self.size = size
        self.pos = body.xpos
        self.xquat = body.xquat
        self.angle = joint_r.qpos

        self._quat: np.ndarray = np.zeros(4)

        self._local_direction: np.ndarray = np.zeros(3)
        self._global_direction: np.ndarray = np.zeros(3)

    def _update_local_direction(self):
        mujoco.mju_axisAngle2Quat(self._quat, [0, 0, 1], self.angle)
        mujoco.mju_rotVecQuat(self._local_direction, [0, 1, 0], self._quat)

    def _update_global_direction(self):
        mujoco.mju_rotVecQuat(self._global_direction, [0, 1, 0], self.xquat)

    def _update(self):
        self._update_local_direction()
        self._update_global_direction()

    @property
    def local_direction(self):
        self.update()
        return self._local_direction

    @property
    def global_direction(self):
        self.update()
        return self._global_direction
