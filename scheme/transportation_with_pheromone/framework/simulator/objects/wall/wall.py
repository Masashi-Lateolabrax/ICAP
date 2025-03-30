from mujoco._structs import _MjDataGeomViews


class Wall:
    def __init__(
            self,
            wall_n: _MjDataGeomViews,
            wall_s: _MjDataGeomViews,
            wall_w: _MjDataGeomViews,
            wall_e: _MjDataGeomViews,
    ):
        self.wall_n = wall_n
        self.wall_s = wall_s
        self.wall_w = wall_w
        self.wall_e = wall_e
