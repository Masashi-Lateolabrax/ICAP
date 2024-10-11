from libs.mujoco_utils.objects import BodyObject

from ._environment import Environment


class Food(BodyObject):
    def __init__(self, env: Environment, food_id: int):
        super().__init__(env.get_model(), env.get_data(), f"food{food_id}")

    def is_valid(self):
        return self.get_body().xpos[2] > 0
