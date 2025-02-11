import mujoco

from mujoco_xml_generator import Sensor, Actuator, Body
from mujoco_xml_generator import common, body, sensor

from scheme.pushing_food_with_pheromone.lib.world import WorldClock

from ...prerude import world
from .food import Food


class FoodBuilder(world.WorldObjectBuilder):
    class NameTable:
        def __init__(self, id_):
            self.BODY = f"food{id_}.body"
            self.JOINT_X = f"food{id_}.joint.x"
            self.JOINT_Y = f"food{id_}.joint.y"
            self.CENTER_SITE = f"food{id_}.site.center"
            self.VELOCIMETER = f"food{id_}.sensor.velocity"

    def __init__(self, id_: int, pos: tuple[float, float], size: float, frictionloss: float):
        super().__init__(f"food{id_}_builder")
        self._name_table = FoodBuilder.NameTable(id_)

        self.pos: tuple[float, float] = pos
        self.size = size
        self.frictionloss = frictionloss

    def gen_body(self) -> Body | None:
        height = 0.07

        return Body(
            name=self._name_table.BODY,
            pos=(self.pos[0], self.pos[1], height)
        ).add_children([
            body.Geom(
                type_=common.GeomType.CYLINDER, size=(self.size, height),
                rgba=(0, 1, 1, 1), density=1000, condim=3
            ),
            body.Joint(
                name=self._name_table.JOINT_Y, type_=common.JointType.SLIDE, axis=(0, 1, 0),
                frictionloss=self.frictionloss
            ),
            body.Joint(
                name=self._name_table.JOINT_X, type_=common.JointType.SLIDE, axis=(0, 1, 0),
                frictionloss=self.frictionloss
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

    def extract(self, model: mujoco.MjModel, data: mujoco.MjData, timer: WorldClock):
        body_ = data.body(self._name_table.BODY)
        velocimeter = data.sensor(self._name_table.VELOCIMETER)
        return Food(body_, velocimeter, timer)
