import os.path

import mujoco
import numpy as np
import cv2

from datetime import datetime
from mujoco_xml_generator.utils.dummy_geom import draw_dummy_geoms

from libs.optimizer import MjcTaskInterface
from ._base import BaseDataCollector


class Recorder(BaseDataCollector):
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
        super().__init__()
        self.episode = episode
        self.width = width
        self.height = height
        self.max_geom = max_geom
        self.camera = camera

        time = datetime.now()
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        self.img = np.zeros((height, width, 3), dtype=np.uint8)
        self.renderer: mujoco.Renderer | None = None

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            os.path.join(project_directory, f"./result_{timestamp}.mp4"),
            fourcc, int(1 / timestep), (width, height)
        )

    def get_episode_length(self) -> int:
        return self.episode

    def pre_record(self, task, time: int):
        pass

    def record(self, task: MjcTaskInterface, time: int, evaluation: float):
        if isinstance(self.renderer, mujoco.Renderer):
            renderer = self.renderer
        else:
            renderer = self.renderer = mujoco.Renderer(
                task.get_model(), self.height, self.width, max_geom=self.max_geom
            )

        self.renderer.update_scene(task.get_data(), self.camera)
        draw_dummy_geoms(task.get_dummies(), renderer)
        renderer.render(out=self.img)
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        self.writer.write(img)

    def release(self):
        print("Saving")
        self.writer.release()
        print("Finish")
