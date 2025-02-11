from mujoco._structs import _MjDataBodyViews, _MjDataSensorViews


class Food:
    def __init__(self, body_: _MjDataBodyViews, velocimeter: _MjDataSensorViews):
        self.body = body_
        self.velocimeter = velocimeter

    @property
    def position(self):
        return self.body.xpos[0:2]

    @property
    def velocity(self):
        return self.velocimeter.data[0:2]
