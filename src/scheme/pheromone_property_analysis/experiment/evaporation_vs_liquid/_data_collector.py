import mujoco
import numpy as np

from lib.utils import BaseDataCollector
from lib.analizer.recorder import Recorder

from .settings import Settings


class DataCollector(BaseDataCollector):
    def __init__(self, working_directory):
        size = Settings.World.NUM_CELL
        episode = int(Settings.Task.EPISODE_LENGTH / Settings.Simulation.PHEROMONE_TIMESTEP + 0.5)
        self.liquid = np.zeros((episode, size[0], size[1]))
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

    def _record(self, task, time: int, evaluation: float):
        self.liquid[time, :, :] = task.pheromone.get_all_liquid()
        self.gas[time, :, :] = task.pheromone.get_all_gas()
        self._recorder._record(task, time, evaluation)
