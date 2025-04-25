import mujoco
from mujoco._structs import _MjDataSiteViews

from libs.mujoco_builder import WorldObjectBuilder, WorldClock
from mujoco_xml_generator import Body, Actuator, Sensor
from mujoco_xml_generator import common, body, actuator
from mujoco_xml_generator.utils import DummyGeom

from ...const import Settings, ROBOT_SIZE, ROBOT_WEIGHT, MOVE_SPEED, TURN_SPEED
from ...sensors import OmniSensor, DirectionSensor
from ..food.name_table import FoodNameTable
from .name_table import RobotNameTable
from .brain import BrainInterface
from .actuator import Actuator as BotActuator
from .robot_input import RobotInput
from .robot_property import RobotProperty

from .robot import Robot


class RobotBuilder(WorldObjectBuilder):
    def __init__(
            self,
            settings: Settings,
            id_: int,
            brain: BrainInterface,
            pos: tuple[float, float, float],
    ):
        super().__init__(f"robot{id_}_builder")
        self.settings = settings

        self.id = id_
        self.brain = brain
        self.name_table = RobotNameTable(id_)
        self.pos = pos

    def gen_static_object(self) -> list[body.Geom | body.Site]:
        return []

    def gen_body(self) -> Body | None:
        return Body(
            name=self.name_table.BODY, pos=(self.pos[0], self.pos[1], 0.06),
            orientation=common.Orientation.AxisAngle(0, 0, 1, self.pos[2])
        ).add_children([
            body.Geom(
                type_=common.GeomType.CYLINDER,
                size=(ROBOT_SIZE, 0.05),
                mass=ROBOT_WEIGHT,
                rgba=(1, 1, 0, 1),
                condim=1
            ),

            body.Joint(
                name=self.name_table.JOINT_X, type_=common.JointType.SLIDE, axis=(1, 0, 0)
            ),
            body.Joint(
                name=self.name_table.JOINT_Y, type_=common.JointType.SLIDE, axis=(0, 1, 0)
            ),
            body.Joint(
                name=self.name_table.JOINT_R, type_=common.JointType.HINGE, axis=(0, 0, 1)
            ),

            body.Site(
                name=self.name_table.FRONT_SITE,
                type_=common.GeomType.SPHERE, size=(0.04,), pos=(0, ROBOT_SIZE - 0.04, 0.051),
                rgba=(1, 0, 0, 1),
            ),
            body.Site(
                name=self.name_table.CENTER_SITE,
                type_=common.GeomType.SPHERE, size=(0.08,), pos=(0, 0, 0.051),
            ),
        ])

    def gen_act(self) -> Actuator | None:
        return Actuator().add_children([
            actuator.Velocity(
                name=self.name_table.ACT_X, joint=self.name_table.JOINT_X, kv=1000
            ),
            actuator.Velocity(
                name=self.name_table.ACT_Y, joint=self.name_table.JOINT_Y, kv=1000
            ),
            actuator.Velocity(
                name=self.name_table.ACT_R, joint=self.name_table.JOINT_R, kv=100,
            )
        ])

    def gen_sen(self) -> Sensor | None:
        return None

    def gen_dummy_geom(self) -> list[DummyGeom]:
        return []

    def extract(self, model: mujoco.MjModel, data: mujoco.MjData, dummy: list[DummyGeom], timer: WorldClock):
        body_ = data.body(self.name_table.BODY)

        property_ = RobotProperty(
            body_,
            ROBOT_SIZE,
            data.joint(self.name_table.JOINT_R),
            timer
        )

        n_robot = self.settings.Robot.NUM
        n_food = self.settings.Food.NUM
        input_ = RobotInput(
            property_,
            other_robot_sensor=OmniSensor(
                self.settings.Robot.OtherRobotSensor.GAIN,
                self.settings.Robot.OtherRobotSensor.OFFSET,
                [data.site(RobotNameTable(i).CENTER_SITE) for i in range(n_robot) if i is not self.id],
            ),
            food_sensor=OmniSensor(
                self.settings.Robot.FoodSensor.GAIN,
                self.settings.Robot.FoodSensor.OFFSET,
                [data.site(FoodNameTable(i).CENTER_SITE) for i in range(n_food)]
            ),
            nest_sensor=DirectionSensor(
                r=model.site("nest").size[0],
                target_site=data.site("nest")
            )
        )

        actuator_ = BotActuator(
            MOVE_SPEED,
            TURN_SPEED,
            property_,
            data.actuator(self.name_table.ACT_X),
            data.actuator(self.name_table.ACT_Y),
            data.actuator(self.name_table.ACT_R),
        )

        center_site: _MjDataSiteViews = data.site(self.name_table.CENTER_SITE)
        center_site_rgba = model.site_rgba[center_site.id]

        return Robot(f"robot{self.id}", self.brain, property_, input_, actuator_, center_site_rgba)
