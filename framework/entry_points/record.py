import mujoco

from libs.utils.data_collector import Recorder

from ..simulator.const import Settings
from ..task import Task
from ..dump import Dump


def record(
        settings: Settings,
        file_path: str,
        task: Task,
) -> Dump | None:
    """
    Record the simulation of an individual.

    Args:
        settings (Settings): The settings object.
        file_path (str): The path to save the recorded video.
        task (framework.Task): The task object to be recorded.

    Returns:
        framework.Dump | None: The dump object if debug is True, otherwise None.
    """

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = settings.Simulation.Render.RENDER_ZOOM

    recorder = Recorder(
        file_path,
        settings.Simulation.TIME_STEP,
        settings.Simulation.TIME_LENGTH,
        settings.Simulation.Render.RENDER_WIDTH,
        settings.Simulation.Render.RENDER_HEIGHT,
        camera,
        Settings.Simulation.Render.MAX_GEOM
    )

    recorder.run(task)
    recorder.release()

    dump = task.get_dump_data()
    return dump if dump is not None else None
