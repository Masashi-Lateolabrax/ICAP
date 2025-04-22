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
        self.robots: dict[str, Robot] = {robot.name: robot for robot in robots}
        self.food: ReFood = refood

        self.dump = Dump() if debug else None

    def get_model(self) -> mujoco.MjModel:
        return self.world.model

    def get_data(self) -> mujoco.MjData:
        return self.world.data

    def _exec_robots(self):
        outputs = {}
        invalid_output = {}
        positions = {}

        for r in self.robots.values():
            r.set_color(0, 1, 0, 1)
            output = r.think()
            r.action(output)
            outputs[r.name] = output
            invalid_output[r.name] = torch.any(torch.isnan(output) | torch.isinf(output))
            positions[r.name] = r.position[:2]

        return outputs, invalid_output, positions

    def calc_step(self) -> float:
        delta = None if self.dump is None else self.dump.create_delta()

        robot_outputs, robot_invalid_output, robot_positions = self._exec_robots()

        if delta is not None:
            delta.robot_outputs = robot_outputs
            delta.robot_pos = robot_positions
            delta.food_pos = self.food.position

        if any(robot_invalid_output.values()):
            for robot_name in filter(lambda x: robot_invalid_output[x], robot_invalid_output.keys()):
                self.robots[robot_name].set_color(1, 0, 0, 1)

            for robot_name, invalid in robot_invalid_output.items():
                if invalid:
                    print(f"Robot {robot_name} has an invalid output (NaN or Inf).")
            print("The output tensor from robots contains invalid values (NaN or Inf).")
            return float("inf")

        robot_pos = np.array([robot.position for robot in self.robots.values()])
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
