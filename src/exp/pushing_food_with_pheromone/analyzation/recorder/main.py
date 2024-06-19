import mujoco

from lib.utils import load_parameter, get_head_hash
from lib.analizer.recorder import Recorder
from src.exp.pushing_food_with_pheromone.settings import HyperParameters, TaskGenerator


def main():
    project_directory = "../../../../"

    task_generator = TaskGenerator()

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = 25

    resolution = 150

    para = load_parameter(
        dim=task_generator.get_dim(),
        working_directory=project_directory,
        git_hash="def49b7a",  # get_head_hash(),
        queue_index=-1,
    )

    task = task_generator.generate(para)
    recorder = Recorder(
        timestep=HyperParameters.Simulator.TIMESTEP,
        episode=int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5),
        width=int(6 * resolution), height=int(9 * resolution),
        project_directory=project_directory,
        camera=camera,
        max_geom=HyperParameters.Simulator.MAX_GEOM,
    )
    recorder.run(task)


if __name__ == '__main__':
    main()
