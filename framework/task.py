import mujoco
import numpy as np
import torch

from libs import optimizer
from libs.mujoco_builder import World

from .simulator.objects.robot import Robot
from .simulator.objects.food import ReFood
from .simulator.objects.nest import Nest
from .simulator.const import Settings

from .dump import Dump


class Task(optimizer.MjcTaskInterface):
    def __init__(
            self,
            settings: Settings,
            para: optimizer.Individual,
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

        self.brain_para = para.view()

        self.dump = Dump() if debug else None

        self.world.calc_step()

    def get_model(self) -> mujoco.MjModel:
        return self.world.model

    def get_data(self) -> mujoco.MjData:
        return self.world.data

    def _exec_robots(self):
        inputs = {}
        outputs = {}
        direction = {}
        invalid_output = {}
        positions = {}

        for r in self.robots.values():
            r.set_color(0, 1, 0, 1)
            output = r.think()
            r.action(output)
            inputs[r.name] = r.input.torch
            outputs[r.name] = output
            direction[r.name] = r.global_direction
            invalid_output[r.name] = torch.any(torch.isnan(output) | torch.isinf(output))
            positions[r.name] = r.position[:2]

        return inputs, outputs, direction, invalid_output, positions

    def calc_step(self) -> float:
        delta = None if self.dump is None else self.dump.create_delta()

        robot_inputs, robot_outputs, robot_direction, robot_invalid_output, robot_positions = self._exec_robots()

        if self.settings.Food.REPLACEMENT:
            self.food.might_replace()

        if delta is not None:
            delta.robot_inputs = {k: v.detach().numpy().copy() for k, v in robot_inputs.items()}
            delta.robot_outputs = {k: v.detach().numpy().copy() for k, v in robot_outputs.items()}
            delta.robot_direction = {k: v.copy() for k, v in robot_direction.items()}
            delta.robot_pos = {k: v.copy() for k, v in robot_positions.items()}
            delta.food_pos = self.food.all_positions()

        if any(robot_invalid_output.values()):
            for robot_name in filter(lambda x: robot_invalid_output[x], robot_invalid_output.keys()):
                self.robots[robot_name].set_color(1, 0, 0, 1)
                print(f"Robot {robot_name} has an invalid output (NaN or Inf).")
            return float("inf")

        robot_pos = np.array([v for v in robot_positions.values()])
        food_pos = self.food.all_positions()
        loss = self.settings.CMAES._loss(self.brain_para, self.nest.position, robot_pos, food_pos)

        self.world.calc_step()

        return loss

    def get_dump_data(self) -> object | None:
        return self.dump

    def run(self) -> float:
        loss = 0
        for _ in range(self.settings.Simulation.TIME_LENGTH):
            loss += self.calc_step()
        return loss
