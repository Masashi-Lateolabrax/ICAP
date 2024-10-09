import mujoco
import numpy as np

from libs.pheromone import PheromoneField
from libs.mujoco_utils.pheromone import PheromoneFieldWithDummies

from ...settings import Settings
from ...utils import robot_names
from ..xml import gen_xml


class Environment:
    def __init__(
            self,
            bot_pos: list[tuple[float, float, float]],
            food_pos: list[tuple[float, float]],
            create_dummies
    ):
        xml = gen_xml(bot_pos, food_pos)

        self._m = mujoco.MjModel.from_xml_string(xml)
        self._d = mujoco.MjData(self._m)

        self.bot_pos = np.zeros((Settings.Task.Robot.NUM_ROBOTS, 2))
        self.food_pos = np.zeros((len(food_pos), 2))
        self.nest_pos = np.array(Settings.Task.Nest.POSITION)

        self.food_nest_dist = np.zeros(len(food_pos))
        self.food_in_nest = np.zeros(len(food_pos), dtype=bool)

        self.pheromone = PheromoneFieldWithDummies(
            PheromoneField(
                nx=Settings.Characteristic.Environment.WIDTH,
                ny=Settings.Characteristic.Environment.HEIGHT,
                d=Settings.Characteristic.Environment.CELL_SIZE,
                sv=Settings.Characteristic.Pheromone.SATURATION_VAPOR,
                evaporate=Settings.Characteristic.Pheromone.EVAPORATION,
                diffusion=Settings.Characteristic.Pheromone.DIFFUSION,
                decrease=Settings.Characteristic.Pheromone.DECREASE
            ),
            cell_size_for_mujoco=Settings.Characteristic.Environment.CELL_SIZE,
            create_dummies=create_dummies,
        )

    def get_model(self):
        return self._m

    def get_data(self):
        return self._d

    def _update_bot_pos(self):
        for bi in range(Settings.Task.Robot.NUM_ROBOTS):
            name_table = robot_names(bi)
            bot_body = self._d.body(name_table["body"])
            self.bot_pos[bi, :] = bot_body.xpos[:2]

    def _update_food_state(self):
        for fi in range(Settings.Task.Food.NUM_FOOD):
            food_body = self._d.body(f"food{fi}")
            self.food_pos[fi, :] = food_body.xpos[:2]

        self.food_nest_dist = np.linalg.norm(self.food_pos - self.nest_pos, axis=1)
        food_in_nest = self.food_nest_dist < Settings.Task.Nest.SIZE
        food_in_nest = np.logical_and(food_in_nest, np.logical_not(self.food_in_nest))

        for fi in np.where(food_in_nest)[0]:
            food_body = self._d.body(f"food{fi}")
            food_body.xpos[2] = Settings.Simulation.CEIL_HEIGHT + 0.071

        self.food_in_nest |= food_in_nest

    def calc_step(self):
        mujoco.mj_step(self._m, self._d)
        self.pheromone.update(
            timestep=Settings.Simulation.TIMESTEP,
            iteration=int(Settings.Simulation.TIMESTEP / Settings.Simulation.Pheromone.TIMESTEP + 0.5),
            dummies=True
        )

        self._update_bot_pos()
        self._update_food_state()

    def get_valid_food(self):
        return self.food_pos[np.logical_not(self.food_in_nest), :]

    def get_dummies(self):
        return self.pheromone.get_dummy_panels()
