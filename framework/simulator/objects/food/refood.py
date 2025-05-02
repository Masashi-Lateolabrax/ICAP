import numpy as np

from ...utils import random_point_avoiding_invalid_areas
from ...const import FOOD_SIZE, NEST_SIZE

from ..nest import Nest
from ..robot import Robot
from .food import Food


class VirtualFood:
    def __init__(self, f: Food):
        self.food: Food = f
        self.copied_pos: np.ndarray | None = None
        self.dummy = False

    def copy(self):
        self.copied_pos = np.copy(self.food.position)
        self.dummy = True
        return VirtualFood(self.food)

    def set_position(self, x, y):
        if self.dummy:
            return
        self.food.joint_x.qpos[0] = x - self.food.joint_x.xanchor[0]
        self.food.joint_y.qpos[0] = y - self.food.joint_y.xanchor[1]
        self.food.position[0:2] = x, y

    def set_velocity_to_zero(self):
        if self.dummy:
            return
        self.food.joint_x.qvel[0] = 0
        self.food.joint_x.qacc[0] = 0
        self.food.joint_y.qvel[0] = 0
        self.food.joint_y.qacc[0] = 0

    @property
    def position(self):
        if self.copied_pos is not None:
            return self.copied_pos
        return self.food.position


def _calc_invalid_area(objs: list[Robot | Nest | Food]) -> list[np.ndarray]:
    invalid_area = []

    for obj in objs:
        area = np.zeros(3)
        area[0:2] = obj.position[0:2]
        area[2] = obj.size
        invalid_area.append(area)

    return invalid_area


def _replace_food(width: float, height: float, invalid_area: list[np.ndarray], padding: float = 0):
    pos = random_point_avoiding_invalid_areas(
        left_upper_point=(-width / 2, height / 2),
        right_lower_point=(width / 2, -height / 2),
        size=FOOD_SIZE,
        invalid_area=invalid_area,
        padding=padding
    )
    return pos


class ReFood:
    """
    The ReFood is a food variant is replaced in the world when it is transported to a nest.
    """

    def __init__(
            self,
            width: float, height: float,
            wall_thickness: float,
            food: list[Food], nest: Nest, robot: list[Robot]
    ):
        self.width = width
        self.height = height
        self.padding = wall_thickness

        self._raw_food = food
        self._food: list[VirtualFood] = [VirtualFood(f) for f in food]

        self._nest = nest
        self._robot = robot

    def __len__(self):
        return len(self._food)

    def might_replace(self):
        invalid_area = _calc_invalid_area(self._robot + [self._nest])
        food_iter = self._food[0:len(self._food)]
        for vf in food_iter:
            if vf.dummy:
                continue

            in_nest = np.linalg.norm(vf.position - self._nest.position) < NEST_SIZE
            if in_nest:
                new_vf = vf.copy()
                new_pos = _replace_food(
                    self.width,
                    self.height,
                    invalid_area + _calc_invalid_area(self._raw_food),
                    padding=self.padding
                )

                new_vf.set_velocity_to_zero()
                new_vf.set_position(*new_pos)
                self._food.append(new_vf)

    def __getitem__(self, item):
        return self._raw_food[item]

    def all_positions(self):
        return np.array([f.position for f in self._food])

    def real_positions(self):
        return np.array([f.position for f in self._raw_food])

    def dummy_positions(self):
        return np.array([f.position for f in self._food if f.dummy])
