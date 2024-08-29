import mujoco


class MujocoObject:
    def __init__(self, model: mujoco.MjModel, type_: mujoco.mjtObj, name: str):
        self.object_name = name
        self.object_id = mujoco.mj_name2id(model, type_, name)