import mujoco

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
        self.pheromone.update(Settings.Simulation.TIMESTEP, 1, True)
        return 0

    def run(self) -> float:
        total_step = int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)
        for _ in range(total_step):
            self.calc_step()
        return 0.0
