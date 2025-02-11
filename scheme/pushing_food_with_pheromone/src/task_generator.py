import numpy as np

import libs.optimizer as opt

from . import utils
from .prerude import Settings
from .task import Task
from .world import World


class TaskGenerator(opt.TaskGenerator):
    @staticmethod
    def _gen_bot_pos(sigma):
        rng = np.random.default_rng()
        bot_pos = np.pad(
            np.array([
                [-0.45, 0.45], [0, 0.45], [0.45, 0.45],
                [-0.45, 0.00], [0, 0.00], [0.45, 0.00],
                [-0.45, -0.45], [0, -0.45], [0.45, -0.45],
            ]),
            ((0, 0), (0, 1)),
            constant_values=90
        )
        bot_pos[:, 2] += sigma * 180 * (2 * rng.random(bot_pos.shape[0]) - 1)
        return bot_pos

    @staticmethod
    def _gen_food_pos(bot_pos):
        w = Settings.Characteristic.Environment.WIDTH_METER
        h = Settings.Characteristic.Environment.HEIGHT_METER

        inviolable_area = [
            np.array([
                Settings.Task.Nest.POSITION[0], Settings.Task.Nest.POSITION[1],
                Settings.Task.REDISTRIBUTION_MARGIN + Settings.Task.Food.SIZE + Settings.Task.Nest.SIZE
            ])
        ]
        for b in bot_pos:
            inviolable_area.append(np.array([
                b[0], b[1],
                Settings.Task.REDISTRIBUTION_MARGIN + Settings.Task.Food.SIZE + 0.175
            ]))

        food_pos = np.zeros((Settings.Task.Food.NUM_FOOD, 2))
        for pos in food_pos:
            pos[:] = utils.random_point_avoiding_invalid_areas(
                (-w * 0.5, h * 0.5),
                (w * 0.5, -h * 0.5),
                inviolable_area
            )
            inviolable_area.append(np.array([
                pos[0], pos[1], Settings.Task.REDISTRIBUTION_MARGIN + Settings.Task.Food.SIZE + Settings.Task.Food.SIZE
            ]))

        return food_pos

    def __init__(self, sigma, panel: bool):
        self.panel = panel
        self.bot_pos = TaskGenerator._gen_bot_pos(sigma)
        self.food_pos = TaskGenerator._gen_food_pos(self.bot_pos)

    @staticmethod
    def get_dim():
        return World.get_dim()

    def generate(self, para, debug=False) -> Task:
        return Task(para, self.bot_pos, self.food_pos, self.panel, debug)
