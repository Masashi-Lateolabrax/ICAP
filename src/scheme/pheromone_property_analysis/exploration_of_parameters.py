import os.path

import numpy as np
import mujoco

from lib.optimizer import CMAES
from lib.analizer.recorder import Recorder

from src.scheme.pheromone_property_analysis.experiment.exploration_of_parameters import *


def main(working_directory):
    generator = Generator()
    cmaes = CMAES(
        dim=5,
        generation=Settings.Optimization.GENERATION,
        population=Settings.Optimization.POPULATION,
        mu=Settings.Optimization.NUM_ELITE,
        sigma=Settings.Optimization.SIGMA,
        split_tasks=1
    )
    cmaes.optimize(generator)
    hist = cmaes.get_history()
    hist.save(os.path.join(working_directory, "history.npy"))

    best = hist.get_min()

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = Settings.Display.ZOOM
    recorder = Recorder(
        timestep=Settings.Simulation.TIMESTEP,
        episode=int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5),
        width=Settings.Display.RESOLUTION[0],
        height=Settings.Display.RESOLUTION[1],
        project_directory=working_directory,
        camera=camera,
        max_geom=Settings.Display.MAX_GEOM
    )
    task = TaskForRec(best.min_para)
    recorder.run(task)
    task.save_log(working_directory)

    para = np.log(1 + np.exp(best.min_para))
    print(f"sv = {para[0]}")
    print(f"evaporate = {para[1]}")
    print(f"diffusion = {para[2]}")
    print(f"decrease = {para[3]}")


if __name__ == '__main__':
    main(os.path.dirname(__file__))
