from array import array
import math

import mujoco
import numpy as np

from libs.optimizer import MjcTaskInterface

from .settings import Settings, EType
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
            bot_pos: list[tuple[float, float, float]],
            food_pos: list[tuple[float, float]],
            panel: bool,
            debug: bool
    ):
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

        self.latest_e = 0
        self.old_e = 0

        self._dump = np.zeros((
            Settings.Simulation.TOTAL_STEP, 2, 2
        ))

    def get_model(self) -> mujoco.MjModel:
        return self.world.env.get_model()

    def get_data(self) -> mujoco.MjData:
        return self.world.env.get_data()

    def calc_step(self) -> float:
        self.world.calc_step()

        self.latest_e = evaluation(
            self._nest_p_sigma,
            self._food_p_sigma,
            self.world.env.food_pos,
            self.world.env.bot_pos,
            self.world.env.nest_pos,
            self._bot_food_dist_buf
        )
        self.old_e = old_evaluation(
            self.init_food_dist,
            self.world.env.food_pos,
            self.world.env.bot_pos,
            self.world.env.nest_pos
        )

        if Settings.Optimization.EVALUATION_TYPE == EType.POTENTIAL:
            e = self.latest_e
        elif Settings.Optimization.EVALUATION_TYPE == EType.DISTANCE:
            e = self.old_e
        else:
            raise Exception("selected invalid EVALUATION_TYPE.")

        return e[0] + e[1]

    def run(self) -> float:
        e = 0
        for t in range(Settings.Simulation.TOTAL_STEP):
            e += self.calc_step()
            self._dump[t, EType.POTENTIAL, :] = self.latest_e
            self._dump[t, EType.DISTANCE, :] = self.old_e
        return e / Settings.Simulation.TOTAL_STEP

    def get_dummies(self):
        return self.world.get_dummies()

    def get_dump_data(self) -> object | None:
        return self._dump
