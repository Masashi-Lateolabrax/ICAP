import mujoco

from lib.utils import load_parameter, get_head_hash
from lib.analizer.recorder import recorder
from src.exp.pushing_food.settings import HyperParameters, TaskGenerator


def main():
    working_directory = "../../../../"

    task_generator = TaskGenerator()

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = 25

    resolution = 150

    para = load_parameter(
        dim=task_generator.get_dim(),
        working_directory=working_directory,
        git_hash=get_head_hash(),
        queue_index=-1,
    )

    task = task_generator.generate(para)
    recorder(
        task=task,
        width=int(6 * resolution), height=int(9 * resolution),
        length=int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5),
        camera=camera,
        timestep=HyperParameters.Simulator.TIMESTEP,
        working_directory=working_directory,
    )


if __name__ == '__main__':
    main()
