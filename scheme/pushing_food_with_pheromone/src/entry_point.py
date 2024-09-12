import os

import numpy as np
import mujoco
import matplotlib.pyplot as plt

from libs.optimizer import CMAES, Hist, MultiThreadProc
from libs.utils.data_collector import Recorder

from .settings import Settings
from .task_generator import TaskGenerator
from .collector import Collector


def optimization() -> Hist:
    dim = TaskGenerator.get_dim()
    print(f"DIM: {dim}")

    cmaes = CMAES(
        dim=dim,
        generation=Settings.Optimization.GENERATION,
        population=Settings.Optimization.POPULATION,
        sigma=Settings.Optimization.SIGMA,
        mu=Settings.Optimization.MU
    )
    for gen in range(1, 1 + cmaes.get_generation()):
        task_generator = TaskGenerator(1, False)
        cmaes.optimize_current_generation(task_generator, MultiThreadProc)

    return cmaes.get_history()


def analysis(work_dir, para):
    total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)

    xs = np.arange(0, total_step) * Settings.Simulation.TIMESTEP

    # Collect data
    generator = TaskGenerator(1, False)
    task = generator.generate(para, True)
    collector = Collector()
    collector.run(task)

    evaluation = collector.evaluation
    pheromone_gas = collector.pheromone_gas

    # Evaluation
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Evaluation")
    axis.plot(xs, evaluation)
    fig.savefig(os.path.join(work_dir, "evaluation.svg"))

    # Pheromone Gas Volume
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Pheromone Gas Volume")
    axis.plot(xs, pheromone_gas)
    fig.savefig(os.path.join(work_dir, "pheromone_gas_volume.svg"))


def record(para, workdir):
    generator = TaskGenerator(1, True)
    task = generator.generate(para, False)

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = 29

    total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)
    rec = Recorder(
        Settings.Simulation.TIMESTEP,
        total_step,
        Settings.Renderer.RESOLUTION[0],
        Settings.Renderer.RESOLUTION[1],
        workdir,
        camera,
        Settings.Renderer.MAX_GEOM
    )

    rec.run(task)
