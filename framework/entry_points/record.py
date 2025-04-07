import mujoco

import framework
from libs import optimizer
from libs.utils.data_collector import Recorder

from ..settings import Settings
from ..interfaceis import BrainBuilder
from ..task_generator import TaskGenerator


def record(
        settings: Settings,
        save_dir,
        para: optimizer.Individual,
        brain_builder: BrainBuilder,
        debug=False
) -> framework.Dump | None:
    """
    Record the simulation of an individual.

    Args:
        settings (Settings): The settings object.
        save_dir (str): The directory to save the recorded data.
        para (optimizer.Individual): The individual to be recorded.
        brain_builder (BrainBuilder): The brain builder object.
        debug (bool): Whether to run in debug mode.

    Returns:
        framework.Dump | None: The dump object if debug is True, otherwise None.
    """

    generator = TaskGenerator(settings, brain_builder)
    task = generator.generate(para, debug=debug)

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = settings.Simulation.RENDER_ZOOM

    recorder = Recorder(
        settings.Simulation.TIME_STEP,
        settings.Simulation.TIME_LENGTH,
        settings.Simulation.RENDER_WIDTH,
        settings.Simulation.RENDER_HEIGHT,
        save_dir,
        camera,
        Settings.Simulation.MAX_GEOM
    )

    recorder.run(task)
    recorder.release()

    return task.get_dump_data() if debug else None
