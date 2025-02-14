import mujoco

from mujoco_xml_generator import Sensor, Actuator, Body
from mujoco_xml_generator import common, body

from scheme.pushing_food_with_pheromone.lib.world import WorldClock, WorldObjectBuilder

from .nest import Nest


class NestBuilder(WorldObjectBuilder):
    def __init__(
            self,
            id_: int,
            pos: tuple[float, float],
            size: float,
            rgba: tuple[float, float, float, float] = (0, 1, 0, 1)
    ):
        super().__init__(f"nest{id_}_builder")

        self.id = id_
        self.pos = pos
        self.size = size
        self.rgba = rgba

    def gen_static_object(self) -> list[body.Geom]:
        thickness = 0.01
        nest_geom = body.Geom(
            name=f"nest{self.id}",
            type_=common.GeomType.CYLINDER,
            size=(self.size, thickness), pos=(self.pos[0], self.pos[1], -thickness - 0.001),
            rgba=self.rgba
        )
        return [nest_geom]

    def gen_body(self) -> Body | None:
        return None

    def gen_act(self) -> Actuator | None:
        return None

    def gen_sen(self) -> Sensor | None:
        return None

    def extract(self, model: mujoco.MjModel, data: mujoco.MjData, timer: WorldClock):
        geom = model.geom(f"nest{self.id}")
        return Nest(geom)
