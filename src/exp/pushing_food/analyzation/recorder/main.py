import mujoco

from lib.utils import load_parameter
from lib.analizer.recorder import recorder
from src.exp.pushing_food.settings import HyperParameters, TaskGenerator


def main():
    task_generator = TaskGenerator()

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = 25

    resolution = 150

    para = load_parameter(
        dim=task_generator.get_dim(),
        load_history_file="",
        queue_index=-1
    )

    task = task_generator.generate(para)
    recorder(
        task=task,
        width=int(6 * resolution), height=int(9 * resolution),
        length=int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5),
        camera=camera,
        timestep=HyperParameters.Simulator.TIMESTEP
    )


if __name__ == '__main__':
    main()
