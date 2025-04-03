import os

import mujoco

from libs import optimizer
from libs.utils.data_collector import Recorder

from ..settings import Settings
from ..interfaceis import BrainBuilder
from ..task_generator import TaskGenerator


def record(settings: Settings, log_path, brain_builder: BrainBuilder):
    logger = optimizer.Hist.load(log_path)
    para = logger._hist.queues[-1].min_para

    generator = TaskGenerator(settings, brain_builder)
    task = generator.generate(para, debug=True)

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = settings.Simulation.RENDER_ZOOM

    recorder = Recorder(
        settings.Simulation.TIME_STEP,
        settings.Simulation.TIME_LENGTH,
        settings.Simulation.RENDER_WIDTH,
        settings.Simulation.RENDER_HEIGHT,
        os.path.dirname(log_path),
        camera,
        Settings.Simulation.MAX_GEOM
    )

    recorder.run(task)
    recorder.release()
