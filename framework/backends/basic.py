import logging
from abc import ABC

import numpy as np
import mujoco

from ..prelude import *
from .utils import render


class BasicMuJoCoSimulator(SimulatorBackend, ABC):
    def __init__(self, settings: Settings, mj_spec: mujoco.MjSpec, render: bool = False):
        self.model: mujoco.MjModel = mj_spec.compile()
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self._max_geom = settings.Render.MAX_GEOM

        self._do_render = render
        self.render_shape = settings.Render.RENDER_WIDTH, settings.Render.RENDER_HEIGHT

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if img_buf is None:
            return
        if not self._do_render:
            logging.warning("Rendering is disabled, skipping render step.")
            return

        render(self.model, self.data, self.render_shape, self._max_geom, img_buf, pos, lookat)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
