import dataclasses

from mujoco._structs import _MjDataActuatorViews

from .robot_property import RobotProperty


@dataclasses.dataclass
class Actuator:
    move_speed: float
    turn_speed: float
    property: RobotProperty
    act_x: _MjDataActuatorViews
    act_y: _MjDataActuatorViews
    act_r: _MjDataActuatorViews

    def turn_left(self):
        self.act_x.ctrl[0] = 0
        self.act_y.ctrl[0] = 0
        self.act_r.ctrl[0] = self.turn_speed

    def turn_right(self):
        self.act_x.ctrl[0] = 0
        self.act_y.ctrl[0] = 0
        self.act_r.ctrl[0] = -self.turn_speed

    def forward(self):
        v = self.property.local_direction * self.move_speed
        self.act_x.ctrl[0] = v[0]
        self.act_y.ctrl[0] = v[1]
        self.act_r.ctrl[0] = 0

    def back(self):
        v = -self.property.local_direction * self.move_speed
        self.act_x.ctrl[0] = v[0]
        self.act_y.ctrl[0] = v[1]
        self.act_r.ctrl[0] = 0

    def stop(self):
        self.act_x.ctrl[0] = 0
        self.act_y.ctrl[0] = 0
        self.act_r.ctrl[0] = 0

    def secretion(self):
        pass
