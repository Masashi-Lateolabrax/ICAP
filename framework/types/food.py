import mujoco


class FoodSpec:
    def __init__(
            self,
            center_site: mujoco._specs.MjsSite,
            free_join: mujoco._specs.MjsJoint,
            velocimeter: mujoco._specs.MjsSensor
    ):
        self.center_site = center_site
        self.free_join = free_join
        self.velocimeter = velocimeter


class FoodValues:
    def __init__(self, data: mujoco.MjData, spec: FoodSpec):
        self._center_site = data.site(spec.center_site.name)
        self._free_join = data.joint(spec.free_join.name)
        self._velocimeter = data.sensor(spec.velocimeter.name)

    @property
    def site(self):
        return self._center_site

    @property
    def xpos(self):
        return self._center_site.xpos[0:2]