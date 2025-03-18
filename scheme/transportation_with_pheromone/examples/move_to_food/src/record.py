import os

import mujoco

from libs.utils.data_collector import Recorder
from libs.optimizer import Hist

from .prerulde import Settings
from .task import TaskGenerator


def record(log_path):
    logger = Hist.load(log_path)
    para = logger._hist.queues[-1].min_para

    generator = TaskGenerator()
    task = generator.generate(para, debug=True)

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = Settings.RENDER_ZOOM

    recorder = Recorder(
        Settings.SIMULATION_TIMESTEP,
        Settings.SIMULATION_TIME_LENGTH,
        Settings.RENDER_WIDTH,
        Settings.RENDER_HEIGHT,
        os.path.dirname(log_path),
        camera,
        Settings.MAX_GEOM
    )

    recorder.run(task)
    recorder.release()
