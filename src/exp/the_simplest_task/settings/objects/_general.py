import mujoco
from mujoco._structs import _MjDataGeomViews, _MjDataBodyViews


class _MujocoObject:
    def __init__(self, model: mujoco.MjModel, type_: mujoco.mjtObj, name: str):
        self.object_name = name
        self.object_id = mujoco.mj_name2id(model, type_, name)


class GeomObject(_MujocoObject):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name: str):
        super().__init__(model, mujoco.mjtObj.mjOBJ_GEOM, name)

        self.geom_id = self.object_id
        self.geom = data.geom(self.object_id)

    def get_geom(self) -> _MjDataGeomViews:
        return self.geom

    def get_geom_id(self) -> int:
        return self.geom_id

    def get_name(self) -> str:
        return self.object_name


class BodyObject(_MujocoObject):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name: str):
        super().__init__(model, mujoco.mjtObj.mjOBJ_BODY, name)

        self.body_id = self.object_id
        self.body = data.body(self.object_id)

    def get_body(self) -> _MjDataBodyViews:
        return self.body

    def get_body_id(self) -> int:
        return self.body_id

    def get_name(self) -> str:
        return self.object_name
