import mujoco

from mujoco_xml_generator import Sensor, Actuator, Body
from mujoco_xml_generator import common, body, sensor
from mujoco_xml_generator.utils import DummyGeom

from libs.mujoco_builder import WorldClock, WorldObjectBuilder

from ...const import FOOD_SIZE, FOOD_DENSITY, FOOD_FRICTIONLOSS

from .name_table import FoodNameTable
from .food import Food


class FoodBuilder(WorldObjectBuilder):
    def __init__(self, id_: int, pos: tuple[float, float]):
        super().__init__(f"food{id_}_builder")
        self._name_table = FoodNameTable(id_)
        self.pos: tuple[float, float] = pos

    def gen_static_object(self) -> list[body.Geom]:
        return []

    def gen_body(self) -> Body | None:
        height = 0.07

        return Body(
            name=self._name_table.BODY,
            pos=(self.pos[0], self.pos[1], height)
        ).add_children([
            body.Geom(
                type_=common.GeomType.CYLINDER, size=(FOOD_SIZE, height),
                rgba=(0, 1, 1, 1), density=FOOD_DENSITY, condim=6
            ),

            body.Joint(
                name=self._name_table.JOINT_Y, type_=common.JointType.SLIDE, axis=(0, 1, 0),
                frictionloss=FOOD_FRICTIONLOSS
            ),
            body.Joint(
                name=self._name_table.JOINT_X, type_=common.JointType.SLIDE, axis=(1, 0, 0),
                frictionloss=FOOD_FRICTIONLOSS
            ),
            body.Joint(
                type_=common.JointType.SLIDE, axis=(0, 0, 1),
                frictionloss=5
            ),

            body.Site(
                name=self._name_table.CENTER_SITE
            )
        ])

    def gen_act(self) -> Actuator | None:
        return None

    def gen_sen(self) -> Sensor | None:
        return Sensor().add_children([
            sensor.Velocimeter(site=self._name_table.CENTER_SITE, name=self._name_table.VELOCIMETER)
        ])

    def gen_dummy_geom(self) -> list[DummyGeom]:
        return []

    def extract(self, model: mujoco.MjModel, data: mujoco.MjData, dummy: list[DummyGeom], timer: WorldClock):
        body_ = data.body(self._name_table.BODY)
        joint_x = data.joint(self._name_table.JOINT_X)
        joint_y = data.joint(self._name_table.JOINT_Y)
        velocimeter = data.sensor(self._name_table.VELOCIMETER)
        return Food(body_, joint_x, joint_y, velocimeter)
