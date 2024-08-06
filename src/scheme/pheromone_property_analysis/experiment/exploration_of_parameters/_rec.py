import os
import warnings

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from mujoco_xml_generator import common as mjc_cmn
from mujoco_xml_generator.utils import DummyGeom

from mujoco_xml_generator import Generator, Option
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import Asset, asset
from mujoco_xml_generator import WorldBody, body

from lib.pheromone import PheromoneField2
from lib.mujoco_utils import PheromoneFieldWithDummies
from lib.optimizer import MjcTaskInterface

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
        para = np.log(1 + np.exp(para))

        self.pheromone = PheromoneFieldWithDummies(
            PheromoneField2(
                nx=Settings.Pheromone.NUM_CELL[0],
                ny=Settings.Pheromone.NUM_CELL[1],
                d=Settings.Pheromone.CELL_SIZE_FOR_CALCULATION,
                sv=para[0],
                evaporate=para[1],
                diffusion=para[2],
                decrease=para[4]
            ),
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
        stability = np.max(np.abs(self.gas_buf[1:] - self.gas_buf[:-1]), axis=(1, 2))

        a = np.where(stability < Settings.Evaluation.STABILITY_THRESHOLD)[0]
        if a.size == 0:
            a = np.array([0])
            warnings.warn("Pheromone viscosity is very high.")
        stable_state_index = np.min(a)
        stable_state_time = stable_state_index * Settings.Simulation.TIMESTEP

        evaporation_speed = self.dif_liquid[stable_state_index]
        gas_volume = np.max(self.gas_buf, axis=(1, 2))
        stable_gas_volume = gas_volume[stable_state_index]
        relative_stable_gas_volume = stable_gas_volume / self.pheromone.get_sv()

        total_step = int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)
        distances = np.ones(total_step) * Settings.Pheromone.CENTER_INDEX[1]
        for t in range(total_step):
            max_gas = self.gas_buf[t, Settings.Pheromone.CENTER_INDEX[0], Settings.Pheromone.CENTER_INDEX[1]]
            sub_gas = self.gas_buf[t, Settings.Pheromone.CENTER_INDEX[0], Settings.Pheromone.CENTER_INDEX[1]:]
            s1 = np.max(np.where(sub_gas >= max_gas * 0.5)[0])
            if s1 == sub_gas.shape[0] - 1:
                break
            g1 = sub_gas[s1]
            g2 = sub_gas[s1 + 1]
            distances[t] = (max_gas * 0.5 - g1) / (g2 - g1) + s1
        distances *= Settings.Pheromone.CELL_SIZE_FOR_CALCULATION
        field_size = np.max(distances)

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(f"The time of stable state: {stable_state_time}")
        axis.plot((1.5 + np.arange(0, stability.shape[0])) * Settings.Simulation.TIMESTEP, stability)
        fig.savefig(os.path.join(working_directory, "stability.svg"))

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(f"The evaporation speed of the stable state: {evaporation_speed}")
        axis.plot((0.5 + np.arange(0, self.dif_liquid.shape[0])) * Settings.Simulation.TIMESTEP, self.dif_liquid)
        fig.savefig(os.path.join(working_directory, "evaporation_speed.svg"))

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(f"absolute: {stable_gas_volume}, relative: {relative_stable_gas_volume}")
        axis.plot((1 + np.arange(0, gas_volume.shape[0])) * Settings.Simulation.TIMESTEP, gas_volume)
        fig.savefig(os.path.join(working_directory, "gas_volume.svg"))

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(f"target: {Settings.Optimization.Loss.FIELD_SIZE:.6f}, size: {field_size:.6f}")
        axis.plot((1 + np.arange(0, distances.shape[0])) * Settings.Simulation.TIMESTEP, distances)
        fig.savefig(os.path.join(working_directory, "size.svg"))
