from mujoco._structs import _MjDataBodyViews, _MjDataSensorViews, _MjDataJointViews


class Food:
    def __init__(
            self,
            body_: _MjDataBodyViews,
            size: float,
            joint_x: _MjDataJointViews,
            joint_y: _MjDataJointViews,
            velocimeter: _MjDataSensorViews
    ):
        self.body = body_
        self.size = size
        self.joint_x = joint_x
        self.joint_y = joint_y
        self.velocimeter = velocimeter

    @property
    def position(self):
        return self.body.xpos[0:2]

    @property
    def velocity(self):
        return self.velocimeter.data[0:2]
