from scheme.pushing_food_with_pheromone.lib.parts import BrainInterface, BrainJudgement


class Brain(BrainInterface):
    @staticmethod
    def get_dim() -> int:
        return 0

    def __init__(self):
        pass

    def think(self) -> BrainJudgement:
        pass
