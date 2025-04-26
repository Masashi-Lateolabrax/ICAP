import os
import dataclasses
from typing import Optional
import tkinter as tk

import mujoco

from mujoco_xml_generator.utils import MuJoCoView

from libs.optimizer import CMAES, Hist, MjcTaskInterface, Individual

from ..prerulde import Settings
from ..task import TaskGenerator
from scheme.transportation_with_pheromone.examples.collect_food.src.robot.brain import Brain


class OptManager:
    @dataclasses.dataclass
    class _Cache:
        renderer: Optional[mujoco.Renderer] = None

    def __init__(self, log_path: str):
        logger = Hist(os.path.dirname(log_path))

        self.cmaes = CMAES(
            dim=Brain.get_dim(),
            generation=Settings.CMAES_GENERATION,
            population=Settings.CMAES_POPULATION,
            mu=Settings.CMAES_MU,
            sigma=Settings.CMAES_SIGMA,
            logger=logger
        )

        self.index = 0
        self.time = 0
        self.best_score = float("inf")

        self.individual: Individual | None = self.cmaes.get_individual(self.index)
        self.task_generator: TaskGenerator = TaskGenerator()
        self.task: MjcTaskInterface | None = self.task_generator.generate(self.individual.view())

        self.individual.fitness.values = (0,)

        self._cache = OptManager._Cache()

    def calc_step(self):
        dt_loss = self.task.calc_step()
        self.individual.fitness.values = (self.individual.fitness.values[0] + dt_loss,)
        self.time += 1

        if self.time >= Settings.SIMULATION_TIME_LENGTH:
            self.time = 0
            self.index = (self.index + 1) % Settings.CMAES_POPULATION

            if self.index == 0:
                num_error, avg, (min_v, min_para), (max_v, max_para), best_para = self.cmaes.update()
                self.cmaes.log(num_error, avg, min_v, min_para, max_v, max_para, best_para)

                self.task_generator = TaskGenerator()

                gen = self.cmaes.get_current_generation()
                self.best_score = min(self.best_score, min_v)
                print(
                    f"finish {gen} gen. error:{num_error}, avg:{avg}, min:{min_v}, max:{max_v}, best:{self.best_score}"
                )

            self.individual = self.cmaes.get_individual(self.index)
            self.individual.fitness.values = (0,)
            self.task = self.task_generator.generate(
                self.individual.view()
            )

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


class App(tk.Tk):
    def __init__(self, log_path):
        super().__init__()
        self.simulation = OptManager(log_path)

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


def test(log_path: str):
    app = App(log_path)
    app.mainloop()
