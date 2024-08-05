import os.path

import mujoco

from lib.optimizer import CMAES
from lib.analizer.recorder import Recorder

if __name__ == "__main__":
    from experiment.exploration_of_parameters import *
else:
    from .experiment.exploration_of_parameters import *


def main(working_directory):
    generator = Generator()
    cmaes = CMAES(
        dim=5,
        generation=Settings.Optimization.GENERATION,
        population=Settings.Optimization.POPULATION,
        mu=Settings.Optimization.NUM_ELITE,
        sigma=Settings.Optimization.SIGMA
    )
    cmaes.optimize(generator)
    hist = cmaes.get_history()
    hist.save(os.path.join(working_directory, "history.npy"))

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
    task = TaskForRec(hist.get_min().min_para)
    recorder.run(task)


if __name__ == '__main__':
    main(os.path.dirname(__file__))
