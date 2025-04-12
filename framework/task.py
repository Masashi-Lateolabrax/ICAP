import mujoco
import numpy as np

from libs import optimizer
from libs.mujoco_builder import World

from .simulator.objects import Nest, Robot, ReFood
from .settings import Settings
from .dump import Dump


class Task(optimizer.MjcTaskInterface):
    def __init__(
            self,
            settings: Settings,
            world: World,
            nest: Nest,
            robots: list[Robot],
            refood: ReFood,
            debug: bool = False
    ):
        self.settings = settings
        self.world: World = world
        self.nest: Nest = nest
        self.robots: list[Robot] = robots
        self.food: ReFood = refood

        self.dump = Dump() if debug else None

    def get_model(self) -> mujoco.MjModel:
        return self.world.model

    def get_data(self) -> mujoco.MjData:
        return self.world.data

    def calc_step(self) -> float:
        for r in self.robots:
            r.exec()
        self.world.calc_step()

        robot_pos = np.array([robot.position for robot in self.robots])
        food_pos = np.array([food.position for food in self.food])
        loss = self.settings.CMAES._loss(self.nest.position, robot_pos, food_pos)

        if isinstance(self.dump, Dump):
            self.dump.dump(robot_pos, food_pos)

        return loss

    def get_dump_data(self) -> object | None:
        return self.dump

    def run(self) -> float:
        loss = 0
        for _ in range(self.settings.Simulation.TIME_LENGTH):
            loss += self.calc_step()
        return loss
