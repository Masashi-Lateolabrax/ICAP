from array import array
import math

import mujoco
import numpy as np

from libs.optimizer import MjcTaskInterface

from .prerude import Settings
from .world import World


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
            bot_pos: np.ndarray,
            food_pos: np.ndarray,
            panel: bool,
            debug: bool
    ):
        self.world = World(para, bot_pos, food_pos, panel, debug)

        self._dump = np.zeros((
            Settings.Simulation.TOTAL_STEP, 2, 2
        ))

    def get_model(self) -> mujoco.MjModel:
        return self.world.env.get_model()

    def get_data(self) -> mujoco.MjData:
        return self.world.env.get_data()

    def calc_step(self) -> float:
        self.world.calc_step()

        vector_between_nest_and_food = self.world.env.nest_pos - self.world.env.food_pos
        v_dist = np.linalg.norm(vector_between_nest_and_food, axis=1)
        normalized_vector = vector_between_nest_and_food / v_dist[:, None]

        food_vel_norm = np.linalg.norm(self.world.env.food_vel, axis=1)
        food_vel_norm = np.where(food_vel_norm == 0, 1, food_vel_norm)
        food_direction = self.world.env.food_vel / food_vel_norm[:, None]

        cos_similarity = np.max(np.sum(normalized_vector * food_direction, axis=1), 0)

        return np.sum(cos_similarity)

    def run(self) -> float:
        e = 0
        for t in range(Settings.Simulation.TOTAL_STEP):
            e += self.calc_step()
        return e / Settings.Simulation.TOTAL_STEP

    def get_dummies(self):
        return self.world.get_dummies()

    def get_dump_data(self) -> object | None:
        return self._dump
