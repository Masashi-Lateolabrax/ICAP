from mujoco._structs import _MjDataSiteViews


class Nest:
    def __init__(self, site: _MjDataSiteViews):
        self.obj = site

    @property
    def position(self):
        return self.obj.xpos[0:2]
