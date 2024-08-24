import os

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from mujoco_xml_generator import common as mjc_cmn
from mujoco_xml_generator.utils import DummyGeom

from mujoco_xml_generator import Generator, Option
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import Asset, asset
from mujoco_xml_generator import WorldBody, body

from lib.mujoco_utils import PheromoneFieldWithDummies
from lib.optimizer import MjcTaskInterface

from ._utils import calc_consistency, init_pheromone_field

from .settings import Settings


def gen_xml():
    resolution = Settings.Display.RESOLUTION

    xml = Generator().add_children([
        Option(
            timestep=Settings.Simulation.TIMESTEP,
            integrator=mjc_cmn.IntegratorType.IMPLICITFACT
        ),
        Visual().add_children([
            visual.Global(offwidth=resolution[0], offheight=resolution[1])
        ]),
        Asset().add_children([
            asset.Texture(
                name="simple_checker", type_=mjc_cmn.TextureType.TWO_DiM, builtin=mjc_cmn.TextureBuiltinType.CHECKER,
                width=100, height=100, rgb1=(1., 1., 1.), rgb2=(0.7, 0.7, 0.7)
            ),
            asset.Material(
                name="ground", texture="simple_checker", texrepeat=(10, 10)
            )
        ]),
        WorldBody().add_children([
            body.Geom(
                type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(0, 0, 1), material="ground"
            ),
        ]),
    ]).build()
    return xml


class TaskForRec(MjcTaskInterface):
    def __init__(self, para):
        self.pheromone = PheromoneFieldWithDummies(
            init_pheromone_field(para),
            Settings.Pheromone.CELL_SIZE_FOR_MUJOCO,
            True
        )

        xml = gen_xml()
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)

        total_step = int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)
        self.t = 0
        self.dif_liquid = np.zeros(total_step)
        self.gas_buf = np.zeros((total_step, *self.pheromone.get_all_gas().shape))

    def get_model(self) -> mujoco.MjModel:
        return self.m

    def get_data(self) -> mujoco.MjData:
        return self.d

    def get_dummies(self) -> list[DummyGeom]:
        return self.pheromone.get_dummy_panels()

    def calc_step(self) -> float:
        center_cell = self.pheromone.get_cell_v2(
            xi=Settings.Pheromone.CENTER_INDEX[0],
            yi=Settings.Pheromone.CENTER_INDEX[1],
        )
        center_cell.set_liquid(Settings.Pheromone.LIQUID)

        before_liquid = self.pheromone.get_all_liquid()
        self.pheromone.update(Settings.Simulation.TIMESTEP, 1, True)

        self.dif_liquid[self.t] = np.sum(self.pheromone.get_all_liquid() - before_liquid) / Settings.Simulation.TIMESTEP
        self.gas_buf[self.t] = self.pheromone.get_all_gas()
        self.t += 1

        return 0

    def run(self) -> float:
        self.t = 0
        self.dif_liquid.fill(0)
        self.gas_buf.fill(0)

        total_step = int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)
        for t in range(total_step):
            self.calc_step()
        return 0.0

    def save_log(self, working_directory):
        sv = self.pheromone.get_sv()

        increase_idx = int(Settings.Optimization.Loss.INCREASE_TIME / Settings.Simulation.TIMESTEP + 0.5)
        increase_time = increase_idx * Settings.Simulation.TIMESTEP

        relative_gas = np.max(self.gas_buf, axis=(1, 2)) / sv

        consistency = calc_consistency(self.gas_buf)

        size_idx = int(Settings.Optimization.Loss.SIZE_TIME / Settings.Simulation.TIMESTEP + 0.5)
        center_idx = Settings.Pheromone.CENTER_INDEX
        size_gas = self.gas_buf[:, center_idx[0], center_idx[1] + Settings.Optimization.Loss.FIELD_SIZE] / sv

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(f"Consistency: {(float(np.min(consistency)) - 1) * 0.5}")
        axis.plot((1 + np.arange(0, consistency.shape[0])) * Settings.Simulation.TIMESTEP, consistency)
        fig.savefig(os.path.join(working_directory, "consistency.svg"))

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(f"The evaporation speed: {self.dif_liquid[-1]}")
        axis.plot((0.5 + np.arange(0, self.dif_liquid.shape[0])) * Settings.Simulation.TIMESTEP, self.dif_liquid)
        fig.savefig(os.path.join(working_directory, "evaporation_speed.svg"))

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(f"time: {increase_time:.6f}, relative: {relative_gas[increase_idx]:.6f}")
        axis.plot((1 + np.arange(0, relative_gas.shape[0])) * Settings.Simulation.TIMESTEP, relative_gas)
        fig.savefig(os.path.join(working_directory, "gas_volume.svg"))

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(f"gas: {size_gas[size_idx]:.6f}")
        axis.plot(np.arange(0, size_gas.shape[0]) * Settings.Simulation.TIMESTEP, size_gas)
        fig.savefig(os.path.join(working_directory, "size.svg"))


class DecTaskForRec(MjcTaskInterface):
    def __init__(self, para):
        self.pheromone = PheromoneFieldWithDummies(
            init_pheromone_field(para),
            Settings.Pheromone.CELL_SIZE_FOR_MUJOCO,
            True
        )

        xml = gen_xml()
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)

        self.total_step = int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)
        self.t = 0
        self.gas_buf = np.zeros((self.total_step, *self.pheromone.get_all_gas().shape))

        self.saturated()

    def saturated(self):
        for xi in range(Settings.Pheromone.NUM_CELL[0]):
            for yi in range(Settings.Pheromone.NUM_CELL[1]):
                cell = self.pheromone.get_cell_v2(xi, yi)
                cell.set_liquid(0.0)
                cell.set_gas(0.0)

        for _ in range(self.total_step):
            center_cell = self.pheromone.get_cell_v2(
                xi=Settings.Pheromone.CENTER_INDEX[0],
                yi=Settings.Pheromone.CENTER_INDEX[1],
            )
            center_cell.set_liquid(Settings.Pheromone.LIQUID)
            self.pheromone.update(Settings.Simulation.TIMESTEP, 1, False)

        self.pheromone.get_cell_v2(
            xi=Settings.Pheromone.CENTER_INDEX[0],
            yi=Settings.Pheromone.CENTER_INDEX[1],
        ).set_liquid(0)

    def get_model(self) -> mujoco.MjModel:
        return self.m

    def get_data(self) -> mujoco.MjData:
        return self.d

    def get_dummies(self) -> list[DummyGeom]:
        return self.pheromone.get_dummy_panels()

    def calc_step(self) -> float:
        self.pheromone.update(Settings.Simulation.TIMESTEP, 1, True)
        self.gas_buf[self.t] = self.pheromone.get_all_gas()
        self.t += 1
        return 0

    def run(self) -> float:
        for _ in range(self.total_step):
            self.calc_step()
        return 0.0

    def save_log(self, working_directory):
        gas_volume = np.max(self.gas_buf, axis=(1, 2))

        # decreased_err
        decrease_idx = int(Settings.Optimization.Loss.DECREASE_TIME / Settings.Simulation.TIMESTEP + 0.5)
        decrease_err = np.max(
            np.abs(self.gas_buf[decrease_idx] - Settings.Optimization.Loss.RELATIVE_STABLE_GAS_VOLUME * 0.5)
        )

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(f"error: {decrease_err}")
        axis.plot(np.arange(0, gas_volume.shape[0]) * Settings.Simulation.TIMESTEP, gas_volume)
        axis.plot(
            decrease_idx * Settings.Simulation.TIMESTEP, Settings.Optimization.Loss.RELATIVE_STABLE_GAS_VOLUME * 0.5
        )
        fig.savefig(os.path.join(working_directory, "gas_volume_dec.svg"))
