import mujoco

from mujoco_xml_generator import Sensor, Actuator, Body
from mujoco_xml_generator import common, body
from mujoco_xml_generator.utils import DummyGeom

from libs.mujoco_builder import WorldClock, WorldObjectBuilder

from ...const import Settings, NEST_SIZE, NEST_THICKNESS
from .nest import Nest


class NestBuilder(WorldObjectBuilder):
    def __init__(
            self,
            settings: Settings,
            rgba: tuple[float, float, float, float] = (0, 1, 0, 1)
    ):
        super().__init__(f"nest_builder")

        self.settings = settings
        self.rgba = rgba

    def gen_static_object(self) -> list[body.Geom | body.Site]:
        pos = self.settings.Nest.POSITION
        nest = body.Site(
            name="nest",
            type_=common.GeomType.CYLINDER,
            size=(NEST_SIZE, NEST_THICKNESS), pos=(pos[0], pos[1], -NEST_THICKNESS - 0.001),
            rgba=self.rgba
        )
        return [nest]

    def gen_body(self) -> Body | None:
        pass

    def gen_act(self) -> Actuator | None:
        pass

    def gen_sen(self) -> Sensor | None:
        pass

    def gen_dummy_geom(self) -> list[DummyGeom]:
        return []

    def extract(self, model: mujoco.MjModel, data: mujoco.MjData, dummy: list[DummyGeom], timer: WorldClock):
        return Nest(data.site("nest"))
