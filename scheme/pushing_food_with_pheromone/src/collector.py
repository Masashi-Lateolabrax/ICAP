import numpy as np

from libs.utils.data_collector import BaseDataCollector

from .settings import Settings
from .task import Task


class Collector(BaseDataCollector):
    def __init__(self):
        self.total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)

        self.pheromone_gas = np.zeros(self.total_step)
        self.evaluation = np.zeros(self.total_step)

    def get_episode_length(self) -> int:
        return self.total_step

    def pre_record(self, task, time: int):
        pass

    def record(self, task: Task, time: int, evaluation: float):
        self.evaluation[time] = evaluation
        self.pheromone_gas[time] = np.max(task.world.env.pheromone.get_all_gas())
