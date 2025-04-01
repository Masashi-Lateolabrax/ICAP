import dataclasses
from typing import Optional

import tkinter as tk
import mujoco

from mujoco_xml_generator.utils import MuJoCoView

import libs
import framework


class SimulationManager:
    @dataclasses.dataclass
    class _Cache:
        renderer: Optional[mujoco.Renderer] = None

    def __init__(self, settings: framework.Settings, task: libs.optimizer.MjcTaskInterface):
        self.settings = settings
        self.task = task
        self._cache = SimulationManager._Cache()

    def draw_on_view_frame(self, view: MuJoCoView, max_geom):
        width = self.settings.Simulation.RENDER_WIDTH
        height = self.settings.Simulation.RENDER_HEIGHT

        if self._cache.renderer is None:
            self._cache.renderer = mujoco.Renderer(
                self.task.get_model(), height, width, max_geom
            )

        view.render(
            self.task.get_data(),
            self._cache.renderer,
            dummy_geoms=self.task.get_dummies()
        )

    def calc_step(self):
        self.task.calc_step()


class App(tk.Tk):
    def __init__(self, settings: framework.Settings, task: libs.optimizer.MjcTaskInterface, max_geom):
        super().__init__()
        self.simulation = SimulationManager(settings, task)
        self.max_geom = max_geom

        self.title("World")

        self.frame = tk.Frame(self)
        self.frame.pack()

        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0, 0, 0]
        self.camera.elevation = -90
        self.camera.distance = settings.Simulation.RENDER_ZOOM

        self.view = MuJoCoView(self.frame, settings.Simulation.RENDER_WIDTH, settings.Simulation.RENDER_HEIGHT)
        self.view.enable_input()
        self.view.pack()
        self.view.camera = self.camera

        self.after(0, self.update)

    def update(self):
        self.simulation.calc_step()
        self.simulation.draw_on_view_frame(self.view, self.max_geom)
        self.after(1, self.update)
