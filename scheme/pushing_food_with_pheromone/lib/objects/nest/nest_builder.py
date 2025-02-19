import mujoco

from mujoco_xml_generator import Sensor, Actuator, Body
from mujoco_xml_generator import common, body

from scheme.pushing_food_with_pheromone.lib.world import WorldClock, WorldObjectBuilder

from .nest import Nest


class NestBuilder(WorldObjectBuilder):
    def __init__(
            self,
            pos: tuple[float, float],
            size: float,
            rgba: tuple[float, float, float, float] = (0, 1, 0, 1)
    ):
        super().__init__(f"nest_builder")

        if type(pos) is not tuple or len(pos) != 2:
            raise ValueError("Invalid argument. 'pos' must be tuple[float, float].")

        self.pos = pos
        self.size = size
        self.rgba = rgba

    def gen_static_object(self) -> list[body.Geom | body.Site]:
        thickness = 0.01
        nest = body.Site(
            name="nest",
            type_=common.GeomType.CYLINDER,
            size=(self.size, thickness), pos=(self.pos[0], self.pos[1], -thickness - 0.001),
            rgba=self.rgba
        )
        return [nest]

    def gen_body(self) -> Body | None:
        return None

    def gen_act(self) -> Actuator | None:
        return None

    def gen_sen(self) -> Sensor | None:
        return None

    def extract(self, model: mujoco.MjModel, data: mujoco.MjData, timer: WorldClock):
        return Nest(
            data.site("nest"),
            self.size
        )
