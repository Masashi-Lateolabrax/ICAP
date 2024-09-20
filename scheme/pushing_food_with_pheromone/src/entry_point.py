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
    generator = TaskGenerator(1, False)
    task = generator.generate(para, True)
    collector = Collector()
    collector.run(task)

    def plot_evaluation():
        total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)
        evaluation = collector.evaluation
        xs = np.arange(0, total_step) * Settings.Simulation.TIMESTEP

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

        axis.plot(xs, evaluation)

        axis.set_title("Evaluation")
        fig.savefig(os.path.join(work_dir, "evaluation.svg"))

    def plot_pheromone_gas_volume():
        total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)
        pheromone_gas = collector.pheromone_gas
        xs = np.arange(0, total_step) * Settings.Simulation.TIMESTEP

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

        axis.plot(xs, pheromone_gas)

        axis.set_title("Pheromone Gas Volume")
        fig.savefig(os.path.join(work_dir, "pheromone_gas_volume.svg"))

    plot_evaluation()
    plot_pheromone_gas_volume()


def analysis2(work_dir, hist: Hist):
    def plot_loss():
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
        fig.savefig(os.path.join(work_dir, "loss.svg"))

    plot_loss()

    analysis(work_dir, hist.get_min().min_para)


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
