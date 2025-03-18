import random

from scheme.transportation_with_pheromone.lib.objects.robot import BrainInterface, BrainJudgement


class Brain(BrainInterface):
    @staticmethod
    def get_dim() -> int:
        return 0

    def __init__(self):
        self.state = BrainJudgement.STOP

    def think(self, input_) -> BrainJudgement:
        r = random.uniform(0, 4 * 200)
        if 0 <= r < 5:
            self.state = BrainJudgement(int(r))
            print(self.state)
        return self.state
