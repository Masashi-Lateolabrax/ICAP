import numpy as np
import mujoco

from libs.utils.data_collector import BaseDataCollector, Recorder

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


class Collector2(BaseDataCollector):
    def __init__(
            self,
            timestep: float,
            episode: int,
            width: int,
            height: int,
            project_directory: str,
            camera: mujoco.MjvCamera,
            max_geom: int = 10000
    ):
        self._recorder = Recorder(timestep, episode, width, height, project_directory, camera, max_geom)
        self.pheromone_gas = np.zeros(episode)
        self.evaluation = np.zeros(episode)

    def get_episode_length(self) -> int:
        return self._recorder.get_episode_length()

    def pre_record(self, task, time: int):
        pass

    def record(self, task, time: int, evaluation: float):
        self._recorder.record(task, time, evaluation)
        self.evaluation[time] = evaluation
        self.pheromone_gas[time] = np.max(task.world.env.pheromone.get_all_gas())

    def release(self):
        self._recorder.release()
