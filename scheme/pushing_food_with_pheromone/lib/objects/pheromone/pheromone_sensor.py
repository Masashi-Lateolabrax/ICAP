from .pheromone import Pheromone


class PheromoneSensor:
    def __init__(self, pheromone: Pheromone):
        self._pheromone = pheromone

    def sense(self, x_pos: float, y_pos: float) -> float:
        return self._pheromone.get(x_pos, y_pos)
