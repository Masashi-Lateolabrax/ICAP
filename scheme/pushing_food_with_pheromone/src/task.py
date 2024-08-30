from array import array
import math

import mujoco
import numpy as np

from libs.optimizer import MjcTaskInterface

from .settings import Settings
from .world import World


class Task(MjcTaskInterface):
    def __init__(self, para: array, panel: bool):
        self.world = World(para, panel)

    def get_model(self) -> mujoco.MjModel:
        return self.world.env.get_model()

    def get_data(self) -> mujoco.MjData:
        return self.world.env.get_data()

    def calc_step(self) -> float:
        self.world.calc_step()

        food_pos = self.world.env.food_pos
        food_vel = self.world.env.food_vel
        bot_pos = self.world.env.bot_pos
        nest_pos = self.world.env.nest_pos

        sub = food_pos - nest_pos
        d = np.linalg.norm(sub, axis=1)
        sub /= d
        food_nest_score = np.dot(food_vel, sub.T)
        food_nest_score = np.sum(food_nest_score)

        dif_food_robot_score = 0
        d = np.zeros((bot_pos.shape[0], 2))
        d[:, 1] = 0.5 + 0.175
        for f in food_pos:
            d[:, 0] = np.linalg.norm(bot_pos - f, axis=1)
            m = np.max(d, axis=1) - d[:, 1]
            dif_food_robot_score -= np.sum(
                np.exp(-m / math.pow(Settings.Optimization.Evaluation.FOOD_RANGE, 2))
            )

        food_nest_score *= Settings.Optimization.Evaluation.FOOD_NEST_GAIN / len(food_pos)
        dif_food_robot_score *= Settings.Optimization.Evaluation.FOOD_ROBOT_GAIN / (len(food_pos) * len(bot_pos))

        # print(dif_food_robot_score, food_nest_score)

        return dif_food_robot_score + food_nest_score

    def run(self) -> float:
        total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)
        evaluation = 0
        for _ in range(total_step):
            evaluation += self.calc_step()
        return evaluation / total_step

    def get_dummies(self):
        return self.world.get_dummies()
