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


def plot_evaluation(work_dir, evaluation):
    total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)
    xs = np.arange(0, total_step) * Settings.Simulation.TIMESTEP

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(xs, evaluation)
    axis.set_title("Evaluation")
    fig.savefig(os.path.join(work_dir, "evaluation.svg"))
    plt.close(fig)


def plot_element_evaluation(work_dir, latest, old):
    xs = np.arange(0, latest.shape[0])
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(xs, latest[:, 0])
    axis.plot(xs, latest[:, 1])
    axis.set_title("Evaluation_latest")
    fig.savefig(os.path.join(work_dir, "evaluation2_latest.svg"))
    plt.close(fig)

    xs = np.arange(0, old.shape[0])
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(xs, old[:, 0])
    axis.plot(xs, old[:, 1])
    axis.set_title("Evaluation_old")
    fig.savefig(os.path.join(work_dir, "evaluation2_old.svg"))
    plt.close(fig)


def plot_pheromone_gas_volume(work_dir, pheromone_gas):
    total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)
    xs = np.arange(0, total_step) * Settings.Simulation.TIMESTEP

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.plot(xs, pheromone_gas)

    axis.set_title("Pheromone Gas Volume")
    fig.savefig(os.path.join(work_dir, "pheromone_gas_volume.svg"))


def plot_loss(workdir, hist):
    xs = np.arange(0, len(hist.queues))
    min_scores = np.zeros(xs.shape[0])
    ave_scores = np.zeros(xs.shape[0])
    max_scores = np.zeros(xs.shape[0])
    for i, q in enumerate(hist.queues):
        min_scores[i] = q.min_score
        max_scores[i] = q.max_score
        ave_scores[i] = q.scores_avg

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.plot(xs, ave_scores)
    axis.fill_between(xs, min_scores, max_scores, color="gray", alpha=0.3)

    axis.set_title("Loss")
    fig.savefig(os.path.join(workdir, "loss.svg"))


def analysis(work_dir, para):
    generator = TaskGenerator(1, False)
    task = generator.generate(para, True)
    collector = Collector()
    collector.run(task)

    plot_evaluation(work_dir, collector.evaluation)
    plot_pheromone_gas_volume(work_dir, collector.pheromone_gas)


# def analysis2(work_dir, hist: Hist):
#     plot_loss(work_dir, hist)
#     analysis(work_dir, hist.get_max().max_para)


def record(para, workdir):
    generator = TaskGenerator(1, True)
    task = generator.generate(para, False)

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = Settings.Renderer.ZOOM

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
