import mujoco

from ._general import GeomObject


class Food(GeomObject):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, food_id: int):
        super().__init__(model, data, f"food{food_id}")
