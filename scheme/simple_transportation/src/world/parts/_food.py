from libs.mujoco_utils.objects import GeomObject

from ._environment import Environment


class Food(GeomObject):
    def __init__(self, env: Environment, food_id: int):
        super().__init__(env.m, env.d, f"food{food_id}")
