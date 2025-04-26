import mujoco
import numpy as np

from ...settings import Settings
from ...utils import robot_names
from ..xml import gen_xml


class Environment:
    def __init__(self):
        xml = gen_xml()

        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)

        self.bot_pos = np.zeros((len(Settings.Task.Robot.POSITIONS), 2))
        self.food_pos = np.zeros((len(Settings.Task.Food.POSITIONS), 2))
        self.nest_pos = np.array(Settings.Task.Nest.POSITION)

    def get_model(self):
        return self.m

    def get_data(self):
        return self.d

    def _update_bot_pos(self):
        for bi in range(len(Settings.Task.Robot.POSITIONS)):
            name_table = robot_names(bi)
            bot_body = self.d.body(name_table["body"])
            self.bot_pos[bi, :] = bot_body.xpos[:2]

    def _update_food_pos(self):
        for fi in range(len(Settings.Task.Food.POSITIONS)):
            geom = self.d.geom(f"food{fi}")
            self.food_pos[fi, :] = geom.xpos[:2]

    def calc_step(self):
        mujoco.mj_step(self.m, self.d)
        self._update_bot_pos()
        self._update_food_pos()
