import mujoco
from mujoco._structs import _MjDataGeomViews

from ._interface import MujocoObject


class GeomObject(MujocoObject):
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
