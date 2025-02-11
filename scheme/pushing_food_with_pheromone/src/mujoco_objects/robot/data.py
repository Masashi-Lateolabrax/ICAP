import numpy as np
import mujoco

from mujoco._structs import _MjDataBodyViews, _MjDataJointViews

from ...prerude import utilities, world


class _CalcCache:
    def __init__(self):
        self.quat: np.ndarray = np.zeros(4)
        self.direction: np.ndarray = np.zeros(3)


class Data(utilities.Cache):
    def __init__(self, body: _MjDataBodyViews, joint_r: _MjDataJointViews, timer: world.WorldClock):
        super().__init__(timer)

        self.pos = body.xpos
        self.angle = joint_r.qpos

        self._cache = _CalcCache()

    def _update_direction(self):
        mujoco.mju_axisAngle2Quat(self._cache.quat, [0, 0, 1], self.angle)
        mujoco.mju_rotVecQuat(self.direction, [0, 1, 0], self._cache.quat)

    def _update(self):
        self._update_direction()

    @property
    def direction(self):
        self.update()
        return self._cache.direction
