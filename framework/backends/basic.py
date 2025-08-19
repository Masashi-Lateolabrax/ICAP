import logging
from abc import ABC

from icecream import ic
import numpy as np
import mujoco

from ..prelude import *


class BasicMuJoCoSimulator(SimulatorBackend, ABC):
    def __init__(self, settings: Settings, mj_spec: mujoco.MjSpec, render: bool = False):
        self.model: mujoco.MjModel = mj_spec.compile()
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self._max_geom = settings.Render.MAX_GEOM

        self._do_render = render
        self.render_shape = settings.Render.RENDER_WIDTH, settings.Render.RENDER_HEIGHT
        self.camera = mujoco.MjvCamera()

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if img_buf is None:
            return
        if not self._do_render:
            logging.warning("Rendering is disabled, skipping render step.")
            return

        try:
            pos = np.array(pos)
            lookat = np.array(lookat)
            sub = pos - lookat
            self.camera.lookat[:] = lookat
            self.camera.distance = np.linalg.norm(sub)
            self.camera.azimuth = np.arctan2(
                sub[1], sub[0]
            ) * 180 / mujoco.mjPI + 180
            self.camera.elevation = -np.arcsin(
                sub[2] / self.camera.distance
            ) * 180 / mujoco.mjPI

            with mujoco.Renderer(
                    self.model, width=self.render_shape[0], height=self.render_shape[1], max_geom=self._max_geom
            ) as renderer:
                renderer.update_scene(self.data, self.camera)
                renderer.render(out=img_buf)

        except Exception as e:
            ic("MuJoCo render error:", e)
            img_buf.fill(0)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
