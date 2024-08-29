from array import array
import math

import mujoco
import numpy as np

from libs.optimizer import MjcTaskInterface

from .settings import Settings
from .world import World


class Task(MjcTaskInterface):
    def __init__(self, para: array):
        self.world = World(para)

    def get_model(self) -> mujoco.MjModel:
        return self.world.env.get_model()

    def get_data(self) -> mujoco.MjData:
        return self.world.env.get_data()

    def calc_step(self) -> float:
        self.world.calc_step()
        food_pos = self.world.env.food_pos
        bot_pos = self.world.env.bot_pos
        nest_pos = self.world.env.nest_pos

        dif_food_nest_score = np.sum(
            np.linalg.norm(food_pos - nest_pos, axis=1)
        )

        dif_food_robot_score = 0
        for f in food_pos:
            d = np.sum((bot_pos - f) ** 2, axis=1)
            dif_food_robot_score -= np.sum(
                np.exp(-d / math.pow(Settings.Optimization.Evaluation.FOOD_RANGE, 2))
            )

        dif_food_nest_score *= Settings.Optimization.Evaluation.FOOD_NEST_GAIN / len(food_pos)
        dif_food_robot_score *= Settings.Optimization.Evaluation.FOOD_ROBOT_GAIN / (len(food_pos) * len(bot_pos))

        return dif_food_robot_score + dif_food_nest_score

    def run(self) -> float:
        evaluation = 0
        for _ in range(int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)):
            evaluation += self.calc_step()
        return evaluation

    def get_dummies(self):
        return []
