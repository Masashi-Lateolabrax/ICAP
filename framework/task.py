import mujoco
import numpy as np
import torch

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
        delta = None if self.dump is None else self.dump.create_delta()

        robot_outputs = {}
        for r in self.robots:
            output = r.think()
            r.action(output)

            robot_outputs[r.name] = output
            if delta is not None:
                delta.robot_pos[r.name] = r.position[:2]

        if delta is not None:
            delta.robot_outputs = robot_outputs
            delta.food_pos = self.food.position

        coned_output = torch.cat(list(robot_outputs.values()), dim=0)
        if torch.any(torch.isnan(coned_output) | torch.isinf(coned_output)):
            print("The output tensor from robots contains invalid values (NaN or Inf).")
            return float("inf")

        robot_pos = np.array([robot.position for robot in self.robots])
        food_pos = np.array([food.position for food in self.food])
        loss = self.settings.CMAES._loss(self.nest.position, robot_pos, food_pos)

        self.world.calc_step()

        return loss

    def get_dump_data(self) -> object | None:
        return self.dump

    def run(self) -> float:
        loss = 0
        for _ in range(self.settings.Simulation.TIME_LENGTH):
            loss += self.calc_step()
        return loss
