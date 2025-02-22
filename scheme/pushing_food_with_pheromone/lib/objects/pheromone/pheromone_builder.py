import mujoco

from mujoco_xml_generator import Sensor, Actuator, Body
from mujoco_xml_generator.utils import DummyGeom

from scheme.pushing_food_with_pheromone.lib.world import WorldClock, WorldObjectBuilder
from .pheromone import Pheromone


class PheromoneBuilder(WorldObjectBuilder):
    def __init__(self, evaporation_speed: float, diffusion_speed: float, near: float, width: int, height: int):
        super().__init__("pheromone_builder")

        self.evaporation_speed = evaporation_speed
        self.diffusion_speed = diffusion_speed
        self.near = near

        self.width = width
        self.height = height

    def gen_body(self) -> Body | None:
        pass

    def gen_act(self) -> Actuator | None:
        pass

    def gen_sen(self) -> Sensor | None:
        pass

    def gen_dummy_geom(self) -> list[DummyGeom] | None:
        pass

    def extract(self, model: mujoco.MjModel, data: mujoco.MjData, dummy: list[DummyGeom], timer: WorldClock):
        return Pheromone(self.evaporation_speed, self.diffusion_speed, self.near, dummy)
