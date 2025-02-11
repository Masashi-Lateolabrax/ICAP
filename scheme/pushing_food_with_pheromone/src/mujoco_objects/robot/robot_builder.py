import mujoco

from mujoco_xml_generator import Sensor, Actuator, Body
from mujoco_xml_generator import common, actuator, body

from scheme.pushing_food_with_pheromone.lib.world import WorldObjectBuilder, WorldClock
from scheme.pushing_food_with_pheromone.lib.parts import OmniSensor, BrainInterface
from ... import utils

from .data import Data
from .actuator import Actuator as BotActuator
from .robot import Robot


class RobotBuilder(WorldObjectBuilder):
    def __init__(
            self,
            id_: int,
            brain: BrainInterface,
            pos: tuple[float, float, float],
            size: float,
            weight: float,
            move_speed: float,
            turn_speed: float,
            sensor_gain: float,
            sensor_offset: float,
            n_food: int = 0
    ):
        super().__init__(f"robot{id_}_builder")
        self.id = id_
        self.brain = brain
        self.name_table = utils.robot_names(id_)
        self.pos = pos
        self.size = size
        self.weight = weight
        self.move_speed = move_speed
        self.turn_speed = turn_speed
        self.sensor_gain = sensor_gain
        self.sensor_offset = sensor_offset
        self.n_food = n_food

    def gen_body(self) -> Body | None:
        return Body(
            name=self.name_table["body"], pos=(self.pos[0], self.pos[1], 0.06),
            orientation=common.Orientation.AxisAngle(0, 0, 1, self.pos[2])
        ).add_children([
            body.Geom(
                type_=common.GeomType.CYLINDER,
                size=(self.size, 0.05),
                mass=self.weight,
                rgba=(1, 1, 0, 0.5)
            ),

            body.Joint(
                name=self.name_table["joint_x"], type_=common.JointType.SLIDE, axis=(1, 0, 0)
            ),
            body.Joint(
                name=self.name_table["joint_y"], type_=common.JointType.SLIDE, axis=(0, 1, 0)
            ),
            body.Joint(
                name=self.name_table["joint_r"], type_=common.JointType.HINGE, axis=(0, 0, 1)
            ),

            body.Site(
                name=self.name_table["front_site"],
                type_=common.GeomType.SPHERE, size=(0.04,), pos=(0, self.size - 0.04, 0.051),
                rgba=(1, 0, 0, 1),
            ),
            body.Site(
                name=self.name_table["center_site"],
                type_=common.GeomType.SPHERE, size=(0.04,)
            ),
        ])

    def gen_act(self) -> Actuator | None:
        return Actuator().add_children([
            actuator.Velocity(
                name=self.name_table["act_x"], joint=self.name_table["joint_x"], kv=1000
            ),
            actuator.Velocity(
                name=self.name_table["act_y"], joint=self.name_table["joint_y"], kv=1000
            ),
            actuator.Velocity(
                name=self.name_table["act_r"], joint=self.name_table["joint_r"], kv=1000
            )
        ])

    def gen_sen(self) -> Sensor | None:
        return None

    def extract(self, model: mujoco.MjModel, data: mujoco.MjData, timer: WorldClock):
        body_ = data.body(self.name_table["body"])
        bot_data = Data(
            body_,
            data.joint(self.name_table["joint_r"]),
            timer
        )
        bot_actuator = BotActuator(
            self.move_speed,
            self.turn_speed,
            bot_data,
            data.actuator(self.name_table["act_x"]),
            data.actuator(self.name_table["act_y"]),
            data.actuator(self.name_table["act_r"]),
        )
        bot_sensor = OmniSensor(
            self.sensor_gain,
            self.sensor_offset,
            [data.site(utils.robot_names(i)["center_site"]) for i in range(self.n_food) if
             i is not self.id]
        )
        return Robot(self.brain, body_, bot_data, bot_actuator, bot_sensor)
