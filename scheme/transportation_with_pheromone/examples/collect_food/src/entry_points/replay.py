import dataclasses
import tkinter as tk
from typing import Optional

import mujoco
from mujoco_xml_generator.utils import MuJoCoView

from libs.optimizer import Hist, MjcTaskInterface

from ..prerulde import Settings
from ..task import TaskGenerator


@dataclasses.dataclass
class _SimCache:
    renderer: Optional[mujoco.Renderer] = None


class SimulationManager:
    def __init__(self, para):
        task_generator = TaskGenerator()
        self.task: MjcTaskInterface = task_generator.generate(para, debug=True)

        self._cache = _SimCache()

    def draw_on_view_frame(self, view: MuJoCoView, max_geom):
        width = Settings.RENDER_WIDTH
        height = Settings.RENDER_HEIGHT

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
    def __init__(self, para):
        super().__init__()
        self.simulation = SimulationManager(para)

        self.title("World")

        self.frame = tk.Frame(self)
        self.frame.pack()

        self.view = MuJoCoView(self.frame, Settings.RENDER_WIDTH, Settings.RENDER_HEIGHT)
        self.view.enable_input()
        self.view.pack()

        self.view.camera.lookat[:] = [0, 0, 0]
        self.view.camera.elevation = -90
        self.view.camera.distance = Settings.RENDER_ZOOM

        self.after(0, self.update)

    def update(self):
        self.simulation.calc_step()
        self.simulation.draw_on_view_frame(self.view, Settings.MAX_GEOM)
        self.after(1, self.update)


def replay(log_path):
    logger = Hist.load(log_path)

    best_para = logger._hist.queues[-1].min_para
    app = App(best_para)
    app.mainloop()
