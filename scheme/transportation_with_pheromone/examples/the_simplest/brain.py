import random

from prelude import *


class Brain(framework.interfaceis.BrainInterface):
    @staticmethod
    def get_dim() -> int:
        return 0

    def __init__(self):
        self.state = framework.BrainJudgement.STOP

    def think(self, input_) -> framework.BrainJudgement:
        r = random.uniform(0, 4 * 200)
        if 0 <= r < 5:
            self.state = framework.BrainJudgement(int(r))
            print(self.state)
        return self.state
