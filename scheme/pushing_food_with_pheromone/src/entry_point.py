import os

import numpy as np
import mujoco
import matplotlib.pyplot as plt

from libs.optimizer import CMAES, MultiThreadProc
from libs.utils.data_collector import Recorder

from .settings import Settings
from .task_generator import TaskGenerator
from .collector import Collector, Collector2
from .logger import Logger


def optimization(workdir):
    dim = TaskGenerator.get_dim()
    print(f"DIM: {dim}")

    logger = Logger(workdir)

    cmaes = CMAES(
        dim=dim,
        generation=Settings.Optimization.GENERATION,
        population=Settings.Optimization.POPULATION,
        sigma=Settings.Optimization.SIGMA,
        mu=Settings.Optimization.MU,
        minimalize=Settings.Optimization.EVALUATION_TYPE != 0,
        logger=logger
    )
    for gen in range(1, 1 + cmaes.get_generation()):
        task_generator = TaskGenerator(1, False)
        cmaes.optimize_current_generation(task_generator, MultiThreadProc)

    logger.save()

    return logger.best_para


def plot_pheromone_gas_volume(work_dir, pheromone_gas):
    total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)
    xs = np.arange(0, total_step) * Settings.Simulation.TIMESTEP

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.plot(xs, pheromone_gas)

    axis.set_title("Pheromone Gas Volume")
    fig.savefig(os.path.join(work_dir, "pheromone_gas_volume.svg"))


def rec_and_collect_data(workdir, para):
    generator = TaskGenerator(1, True)
    task = generator.generate(para, False)

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = Settings.Renderer.ZOOM

    total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)
    collector = Collector2(
        Settings.Simulation.TIMESTEP,
        total_step,
        Settings.Renderer.RESOLUTION[0],
        Settings.Renderer.RESOLUTION[1],
        workdir,
        camera,
        Settings.Renderer.MAX_GEOM
    )

    collector.run(task)
    collector.release()

    return collector.evaluation, collector.pheromone_gas


def sampling(workdir, para):
    evaluation, pheromone_gas = rec_and_collect_data(workdir, para)
    plot_evaluation(workdir, evaluation)
    plot_pheromone_gas_volume(workdir, pheromone_gas)
