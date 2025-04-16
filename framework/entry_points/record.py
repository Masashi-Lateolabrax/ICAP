import os
import mujoco

import framework
from libs import optimizer
from libs.utils.data_collector import Recorder

from ..settings import Settings
from ..simulator.objects import BrainBuilder
from ..task_generator import TaskGenerator


def record(
        settings: Settings,
        file_path: str,
        para: optimizer.Individual,
        brain_builder: BrainBuilder,
        debug=False
) -> framework.Dump | None:
    """
    Record the simulation of an individual.

    Args:
        settings (Settings): The settings object.
        file_path (str): The path to save the recorded video.
        para (optimizer.Individual): The individual to be recorded.
        brain_builder (BrainBuilder): The brain builder object.
        debug (bool): Whether to run in debug mode.

    Returns:
        framework.Dump | None: The dump object if debug is True, otherwise None.
    """

    save_dir = os.path.dirname(file_path)
    if os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    generator = TaskGenerator(settings, brain_builder)
    task = generator.generate(para, debug=debug)

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = settings.Simulation.RENDER_ZOOM

    recorder = Recorder(
        file_path,
        settings.Simulation.TIME_STEP,
        settings.Simulation.TIME_LENGTH,
        settings.Simulation.RENDER_WIDTH,
        settings.Simulation.RENDER_HEIGHT,
        camera,
        Settings.Simulation.MAX_GEOM
    )

    recorder.run(task)
    recorder.release()

    return task.get_dump_data() if debug else None
