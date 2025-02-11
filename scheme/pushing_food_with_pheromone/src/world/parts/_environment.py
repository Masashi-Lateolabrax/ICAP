import mujoco
import numpy as np

from libs.pheromone import PheromoneField
from libs.mujoco_utils.pheromone import PheromoneFieldWithDummies

from ... import utils
from ...prerude import Settings
from ..xml import gen_xml


class Environment:
    def __init__(
            self,
            bot_pos: np.ndarray,
            food_pos: np.ndarray,
            create_dummies
    ):
        xml = gen_xml(bot_pos, food_pos)

        self._m = mujoco.MjModel.from_xml_string(xml)
        self._d = mujoco.MjData(self._m)

        self.bot_pos = np.zeros((Settings.Task.Robot.NUM_ROBOTS, 2))
        self.food_pos = np.zeros((len(food_pos), 2))
        self.food_vel = np.zeros((len(food_pos), 2))
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

    def _update_bot_buf(self):
        for bi in range(Settings.Task.Robot.NUM_ROBOTS):
            name_table = utils.robot_names(bi)
            bot_body = self._d.body(name_table["body"])
            self.bot_pos[bi, :] = bot_body.xpos[:2]

    def _update_food_buf(self):
        for fi in range(Settings.Task.Food.NUM_FOOD):
            food_body = self._d.body(f"food{fi}")
            food_x_joint = self._d.joint(f"food{fi}.joint.slide_x")
            food_y_joint = self._d.joint(f"food{fi}.joint.slide_y")
            self.food_pos[fi, :] = pos = food_body.xpos[:2]
            self.food_vel[fi, 0] = food_x_joint.qvel[0]
            self.food_vel[fi, 1] = food_y_joint.qvel[0]
            self.food_nest_dist[fi] = dist = np.linalg.norm(pos - self.nest_pos)
            self.food_in_nest[fi] = dist < Settings.Task.Nest.SIZE

    def _redistribute_food(self):
        invalid_area = [
            np.array([
                self.nest_pos[0], self.nest_pos[1],
                Settings.Task.REDISTRIBUTION_MARGIN + Settings.Task.Food.SIZE + Settings.Task.Nest.SIZE
            ])
        ]
        for b in self.bot_pos:
            invalid_area.append(np.array([
                b[0], b[1],
                Settings.Task.REDISTRIBUTION_MARGIN + Settings.Task.Food.SIZE + 0.175
            ]))
        for f in self.food_pos:
            invalid_area.append(np.array([
                f[0], f[1],
                Settings.Task.REDISTRIBUTION_MARGIN + Settings.Task.Food.SIZE + Settings.Task.Food.SIZE
            ]))

        w = Settings.Characteristic.Environment.WIDTH_METER
        h = Settings.Characteristic.Environment.HEIGHT_METER
        for fi, _ in filter(lambda x: x[1], enumerate(self.food_in_nest)):
            pos = utils.random_point_avoiding_invalid_areas(
                (-w * 0.5, h * 0.5),
                (w * 0.5, -h * 0.5),
                invalid_area,
                retry=5
            )
            if pos is not None:
                food_x_joint = self._d.joint(f"food{fi}.joint.slide_x")
                food_y_joint = self._d.joint(f"food{fi}.joint.slide_y")
                food_x_joint.qvel[0] = 0
                food_y_joint.qvel[0] = 0
                food_x_joint.qpos[0] = pos[0]
                food_y_joint.qpos[0] = pos[1]
                self.food_in_nest[fi] = False

    def calc_step(self):
        mujoco.mj_step(self._m, self._d)
        self.pheromone.update(
            timestep=Settings.Simulation.TIMESTEP,
            iteration=int(Settings.Simulation.TIMESTEP / Settings.Simulation.Pheromone.TIMESTEP + 0.5),
            dummies=True
        )

        self._update_bot_buf()
        self._update_food_buf()
        self._redistribute_food()

    def get_dummies(self):
        return self.pheromone.get_dummy_panels()
