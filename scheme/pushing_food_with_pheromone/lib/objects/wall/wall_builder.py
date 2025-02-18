import mujoco

from mujoco_xml_generator import Sensor, Actuator, Body
from mujoco_xml_generator import common, body

from scheme.pushing_food_with_pheromone.lib.world import WorldObjectBuilder, WorldClock

from .wall import Wall


class WallBuilder(WorldObjectBuilder):
    def __init__(
            self,
            world_width: float,
            world_height: float,
            thickness: float,
            wall_height: float = 0.1
    ):
        super().__init__(f"wall_builder")
        self.world_width = world_width
        self.world_height = world_height
        self.thickness = thickness
        self.wall_height = wall_height

    def gen_static_object(self) -> list[body.Geom]:
        res = []
        w = self.world_width + self.thickness * 2
        h = self.world_height + self.thickness * 2
        for name, x, y, ww, wh in [
            ("wallN", 0, h * 0.5, w * 0.5, self.thickness),
            ("wallS", 0, h * -0.5, w * 0.5, self.thickness),
            ("wallW", w * 0.5, 0, self.thickness, h * 0.5),
            ("wallE", w * -0.5, 0, self.thickness, h * 0.5),
        ]:
            res.append(body.Geom(
                name=name, type_=common.GeomType.BOX,
                pos=(x, y, self.wall_height), size=(ww, wh, self.wall_height),
                condim=1
            ))
        return res

    def gen_body(self) -> Body | None:
        return None

    def gen_act(self) -> Actuator | None:
        return None

    def gen_sen(self) -> Sensor | None:
        return None

    def extract(self, model: mujoco.MjModel, data: mujoco.MjData, timer: WorldClock):
        wall_n = model.geom("wallN")
        wall_s = model.geom("wallS")
        wall_w = model.geom("wallW")
        wall_e = model.geom("wallE")
        return Wall(wall_n, wall_s, wall_w, wall_e)
