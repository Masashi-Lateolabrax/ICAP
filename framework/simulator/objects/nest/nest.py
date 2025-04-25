from mujoco._structs import _MjDataSiteViews

from ...const import NEST_SIZE


class Nest:
    size = NEST_SIZE

    def __init__(self, site: _MjDataSiteViews):
        self.obj = site

    @property
    def position(self):
        return self.obj.xpos[0:2]
