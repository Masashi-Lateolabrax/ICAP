import dataclasses
from typing import Optional

import tkinter as tk

import mujoco

from mujoco_xml_generator.utils import MuJoCoView

from scheme.pushing_food_with_pheromone.lib import world
from scheme.pushing_food_with_pheromone.src.mujoco_objects import robot, food

from settings import Settings


@dataclasses.dataclass
class _SimCache:
    renderer: Optional[mujoco.Renderer] = None


class SimulationManager:
    def __init__(self, world_: world.World, robot_: robot.Robot, food_: food.Food):
        self.world = world_
        self.robot = robot_
        self.food = food_

        self._cache = _SimCache()

    def draw_on_view_frame(self, view: MuJoCoView, max_geom):
        width = Settings.RENDER_WIDTH
        height = Settings.RENDER_HEIGHT

        if self._cache.renderer is None:
            self._cache.renderer = mujoco.Renderer(
                self.world.model, height, width, max_geom
            )

        view.render(
            self.world.data,
            self._cache.renderer,
            dummy_geoms=self.world.get_dummy_geoms()
        )

    def calc_step(self):
        self.robot.action()
        self.world.calc_step()


class App(tk.Tk):
    def __init__(self, simulation: SimulationManager, width, height, max_geom):
        super().__init__()
        self.simulation = simulation
        self.max_geom = max_geom

        self.title("World")

        self.frame = tk.Frame(self)
        self.frame.pack()

        self.view = MuJoCoView(self.frame, width, height)
        self.view.enable_input()
        self.view.pack()

        self.after(0, self.update)

    def update(self):
        self.simulation.calc_step()
        self.simulation.draw_on_view_frame(self.view, self.max_geom)
        self.after(1, self.update)
