from array import array
import math

import mujoco
import numpy as np

from libs.optimizer import MjcTaskInterface

from .settings import Settings
from .world import World
from ..main import LogFragment


def _squared_sigma(size, p):
    return np.log10(np.exp((size ** 2) / p))


def evaluation(nest_p_sigma, food_p_sigma, food_pos, bot_pos, nest_pos, bot_food_dist_buf):
    dist_bet_n_f = np.linalg.norm(food_pos - nest_pos, axis=1)
    nest_food_score = np.sum(np.exp(-(dist_bet_n_f ** 2) / nest_p_sigma))

    food_robot_score = 0
    for f in food_pos:
        bot_food_dist_buf[:, 0] = np.linalg.norm(bot_pos - f, axis=1)
        m = np.max(bot_food_dist_buf, axis=1) - bot_food_dist_buf[:, 1]
        food_robot_score += np.sum(np.exp(-(m ** 2) / food_p_sigma))

    nest_food_score *= Settings.Optimization.Evaluation.NEST_GAIN / len(food_pos)
    food_robot_score *= Settings.Optimization.Evaluation.FOOD_GAIN / (len(food_pos) * len(bot_pos))

    return food_robot_score, nest_food_score


def old_evaluation(init_food_dist, food_pos, bot_pos, nest_pos):
    food_nest_score = np.sum(
        np.linalg.norm(food_pos - nest_pos, axis=1) / init_food_dist
    )

    dif_food_robot_score = 0
    d = np.zeros((bot_pos.shape[0], 2))
    d[:, 1] = 0.5 + 0.175
    for f in food_pos:
        d[:, 0] = np.linalg.norm(bot_pos - f, axis=1)
        m = np.max(d, axis=1) - d[:, 1]
        dif_food_robot_score -= np.sum(
            np.exp(-m / math.pow(Settings.Optimization.OldEvaluation.FOOD_RANGE, 2))
        )

    food_nest_score *= Settings.Optimization.OldEvaluation.FOOD_NEST_GAIN / len(food_pos)
    dif_food_robot_score *= Settings.Optimization.OldEvaluation.FOOD_ROBOT_GAIN / (len(food_pos) * len(bot_pos))

    return dif_food_robot_score, food_nest_score


class Task(MjcTaskInterface):
    def __init__(
            self,
            para: array,
            bot_pos: list[tuple[float, float, float]],
            food_pos: list[tuple[float, float]],
            panel: bool,
            logger,
            debug: bool
    ):
        self.log_fragment = LogFragment(para)
        self.logger = logger

        self.world = World(para, bot_pos, food_pos, panel, debug)
        self.init_food_dist = np.linalg.norm(np.array(food_pos), axis=1)

        self._food_p_sigma = _squared_sigma(
            Settings.Optimization.Evaluation.FOOD_RANGE, Settings.Optimization.Evaluation.FOOD_RANGE_P
        )
        self._nest_p_sigma = _squared_sigma(
            self.init_food_dist, Settings.Optimization.Evaluation.NEST_RANGE_P
        )

        self._bot_food_dist_buf = np.zeros((Settings.Task.Robot.NUM_ROBOTS, 2))
        self._bot_food_dist_buf[:, 1] = 0.5 + 0.175

    def get_model(self) -> mujoco.MjModel:
        return self.world.env.get_model()

    def get_data(self) -> mujoco.MjData:
        return self.world.env.get_data()

    def calc_step(self) -> float:
        self.world.calc_step()

        e_latest = evaluation(
            self._nest_p_sigma,
            self._food_p_sigma,
            self.world.env.food_pos,
            self.world.env.bot_pos,
            self.world.env.nest_pos,
            self._bot_food_dist_buf
        )
        e_old = old_evaluation(
            self.init_food_dist,
            self.world.env.food_pos,
            self.world.env.bot_pos,
            self.world.env.nest_pos
        )

        self.log_fragment.add_score((e_latest, e_old))

        return e_old[0] + e_old[1]

    def run(self) -> float:
        total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)
        e = 0
        for _ in range(total_step):
            e += self.calc_step()
        if self.logger is not None:
            self.logger.add_fragment(self.log_fragment)
        return e / total_step

    def get_dummies(self):
        return self.world.get_dummies()
