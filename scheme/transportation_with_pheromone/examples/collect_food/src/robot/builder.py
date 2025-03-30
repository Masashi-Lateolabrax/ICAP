from mujoco_xml_generator import Sensor, Actuator, Body
from mujoco_xml_generator import common, actuator, body
from mujoco_xml_generator.utils import DummyGeom

from scheme.transportation_with_pheromone.lib.world import WorldObjectBuilder, WorldClock

from scheme.transportation_with_pheromone.lib.objects import robot

from .brain import Brain, BrainJudgement


class RobotBuilder(robot.RobotBuilder):
    def extract(self, model: mujoco.MjModel, data: mujoco.MjData, dummy: list[DummyGeom], timer: WorldClock):
        body_ = data.body(self.name_table.BODY)

        property_ = RobotProperty(
            body_,
            self.size,
            data.joint(self.name_table.JOINT_R),
            timer
        )

        input_ = RobotInput(
            property_,
            other_robot_sensor=OmniSensor(
                self.sensor_gain,
                self.sensor_offset,
                [data.site(RobotNameTable(i).CENTER_SITE) for i in range(self.n_food) if i is not self.id],
            ),
            food_sensor=OmniSensor(
                self.sensor_gain,
                self.sensor_offset,
                [data.site(FoodNameTable(i).CENTER_SITE) for i in range(self.n_food)]
            ),
            nest_sensor=DirectionSensor(
                r=model.site("nest").size[0],
                target_site=data.site("nest")
            )
        )

        actuator_ = BotActuator(
            self.move_speed,
            self.turn_speed,
            property_,
            data.actuator(self.name_table.ACT_X),
            data.actuator(self.name_table.ACT_Y),
            data.actuator(self.name_table.ACT_R),
        )

        return Robot(self.brain, property_, input_, actuator_)
