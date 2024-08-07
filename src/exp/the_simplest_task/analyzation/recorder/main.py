import mujoco

from lib.utils import load_parameter, get_head_hash
from lib.analizer.recorder import recorder
from src.exp.the_simplest_task.settings import HyperParameters, TaskGenerator


def main():
    project_directory = "../../../../"
    git_hash = get_head_hash()

    task_generator = TaskGenerator()

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = 25

    resolution = 150

    para = load_parameter(
        dim=task_generator.get_dim(),
        working_directory=project_directory,
        git_hash=git_hash,
        queue_index=-1,
    )

    task = task_generator.generate(para)
    recorder(
        task=task,
        width=int(6 * resolution), height=int(9 * resolution),
        length=int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5),
        camera=camera,
        timestep=HyperParameters.Simulator.TIMESTEP,
        working_directory=project_directory,
        max_geom=HyperParameters.Simulator.MAX_GEOM,
    )


if __name__ == '__main__':
    main()
