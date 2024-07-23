import mujoco

from lib.utils import load_parameter, get_head_hash
from lib.analizer.recorder import Recorder

from settings import GeneralSetting, TaskGenerator


def recorder():
    project_directory = "../../../../"
    git_hash = get_head_hash()
    generation = -1

    task_generator = TaskGenerator()

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = 29

    resolution = 150

    para = load_parameter(
        dim=task_generator.get_dim(),
        working_directory=project_directory,
        git_hash=git_hash,
        queue_index=generation,
    )

    task = task_generator.generate(para)
    rec = Recorder(
        timestep=GeneralSetting.Simulation.TIMESTEP,
        episode=int(GeneralSetting.Task.EPISODE_LENGTH / GeneralSetting.Simulation.TIMESTEP + 0.5),
        width=int(6 * resolution), height=int(9 * resolution),
        project_directory=project_directory,
        camera=camera,
        max_geom=GeneralSetting.Display.MAX_GEOM,
    )
    rec.run(task)


if __name__ == '__main__':
    recorder()
