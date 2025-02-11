import numpy as np
from mujoco._structs import _MjDataBodyViews, _MjDataSensorViews

from ...prerude import world


class Data:
    def __init__(self, body_: _MjDataBodyViews, velocimeter: _MjDataSensorViews, timer: world.WorldClock):
        self._update_time = None
        self._timer = timer

        self._body = body_
        self._velocimeter = velocimeter

        self.position = np.zeros(2)
        self.velocity = np.zeros(2)

    def update(self):
        t = self._timer.get()
        if self._update_time == t:
            return
        self._update_time = t

        self.position[:] = self._body.xpos[0:2]
        self.velocity[:] = self._velocimeter.data[0:2]


class Food:
    def __init__(self, body_: _MjDataBodyViews, velocimeter: _MjDataSensorViews, timer: world.WorldClock):
        self.data = Data(body_, velocimeter, timer)

    def calc_step(self):
        self.data.update()

    @property
    def position(self):
        return self.data.position

    @property
    def velocity(self):
        return self.data.velocity
