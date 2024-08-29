import mujoco
from mujoco._structs import _MjDataBodyViews

from ._interface import MujocoObject


class BodyObject(MujocoObject):
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
