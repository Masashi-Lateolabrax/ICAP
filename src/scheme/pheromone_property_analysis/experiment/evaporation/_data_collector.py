import mujoco
import numpy as np

from lib.utils import BaseDataCollector
from lib.analizer.recorder import Recorder

from .settings import Settings
from ._task import AnalysisEnvironment


class DataCollector(BaseDataCollector):
    def __init__(self, working_directory):
        size = Settings.World.NUM_CELL
        episode = int(Settings.Task.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)

        self.dif_liquid = np.zeros(episode)
        self.gas = np.zeros((episode, size[0], size[1]))

        camera = mujoco.MjvCamera()
        camera.elevation = -90
        camera.distance = Settings.Display.ZOOM

        self._recorder = Recorder(
            timestep=Settings.Simulation.TIMESTEP,
            episode=episode,
            width=Settings.Display.RESOLUTION[0],
            height=Settings.Display.RESOLUTION[1],
            project_directory=working_directory,
            camera=camera,
            max_geom=Settings.Display.MAX_GEOM
        )

    def get_episode_length(self) -> int:
        return self._recorder.get_episode_length()

    def pre_record(self, task, time: int):
        self._recorder.pre_record(task, time)

    def record(self, task: AnalysisEnvironment, time: int, evaluation: float):
        self.dif_liquid[time] = task.dif_liquid
        self.gas[time, :, :] = task.pheromone.get_all_gas()
        self._recorder.record(task, time, evaluation)

    def release(self):
        self._recorder.release()
