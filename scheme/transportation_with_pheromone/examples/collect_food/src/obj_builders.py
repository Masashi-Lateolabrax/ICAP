import numpy as np

from scheme.transportation_with_pheromone.lib.utilities import random_point_avoiding_invalid_areas
from scheme.transportation_with_pheromone.lib.objects.robot import RobotBuilder, BrainInterface, RobotInput, \
    BrainJudgement
from scheme.transportation_with_pheromone.lib.objects.food import FoodBuilder

from .prerulde import Settings


class _DummyBrain(BrainInterface):
    @staticmethod
    def get_dim() -> int:
        return 0

    def think(self, input_: RobotInput) -> BrainJudgement:
        return BrainJudgement.STOP


