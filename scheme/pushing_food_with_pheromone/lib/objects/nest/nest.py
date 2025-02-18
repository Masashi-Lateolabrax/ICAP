from mujoco._structs import _MjDataGeomViews


class Nest:
    def __init__(self, gome: _MjDataGeomViews, size: float):
        self.size = size
        self.geom = gome

    @property
    def position(self):
        return self.geom.xpos[0:2]
