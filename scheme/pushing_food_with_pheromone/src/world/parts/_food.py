from libs.mujoco_utils.objects import GeomObject

from ._environment import Environment


class Food(GeomObject):
    def __init__(self, env: Environment, food_id: int):
        super().__init__(env.get_model(), env.get_data(), f"food{food_id}")

    def is_valid(self):
        return self.get_geom().xpos[2] > 0
