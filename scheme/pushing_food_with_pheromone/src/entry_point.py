import os

import numpy as np
import mujoco
import matplotlib.pyplot as plt

from libs.optimizer import CMAES, MultiThreadProc

from .settings import Settings, EType
from .task_generator import TaskGenerator
from .collector import Collector2
from .logger import Logger, LogLoader


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
    xs = np.arange(0, Settings.Simulation.TOTAL_STEP) * Settings.Simulation.TIMESTEP

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(xs, evaluation)
    axis.set_title("Evaluation")
    fig.savefig(os.path.join(work_dir, "evaluation.svg"))
    plt.close(fig)


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

    collector = Collector2(
        Settings.Simulation.TIMESTEP,
        Settings.Simulation.TOTAL_STEP,
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


def plot_evaluation_elements_for_each_generation(workdir, loader: LogLoader):
    evaluations = np.zeros((Settings.Optimization.GENERATION, 3))

    for gen in range(len(loader)):
        inds = loader.get_individuals(gen)

        dump = np.array([i.dump for i in inds])
        dump = np.sum(dump, axis=1)
        summed_dump = np.sum(dump, axis=2)

        if Settings.Optimization.EVALUATION_TYPE == EType.POTENTIAL:
            i = np.argmax(summed_dump[:, EType.POTENTIAL])
            evaluations[gen, 0:2] = dump[i, EType.POTENTIAL, :]
            evaluations[gen, 2] = np.sum(evaluations[gen, 0:2])

        elif Settings.Optimization.EVALUATION_TYPE == EType.DISTANCE:
            i = np.argmin(summed_dump[:, EType.DISTANCE])
            evaluations[gen, 0:2] = dump[i, EType.DISTANCE, :]
            evaluations[gen, 2] = np.sum(evaluations[gen, 0:2])

        else:
            raise Exception("Selected an invalid EVALUATION_TYPE.")

    evaluations /= Settings.Simulation.TOTAL_STEP
    xs = np.arange(0, evaluations.shape[0])

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(xs, evaluations[:, 2], c="#d3d3d3")
    axis.plot(xs, evaluations[:, 0], c="#0000ff")
    axis.plot(xs, evaluations[:, 1], c="#ff0000")

    fig.savefig(
        os.path.join(workdir, "evaluation_elements_for_each_generation.svg")
    )
    plt.close(fig)


def plot_evaluation_for_each_generation(workdir, loader: LogLoader):
    e_type = Settings.Optimization.EVALUATION_TYPE
    evaluations = np.zeros((Settings.Optimization.GENERATION, 3))

    for gen in range(len(loader)):
        inds = loader.get_individuals(gen)

        dump = np.array([i.dump for i in inds])
        dump = np.sum(dump, axis=1)
        summed_dump = np.sum(dump, axis=2)

        min_i = np.argmin(summed_dump[:, e_type])
        min_score = summed_dump[min_i, e_type]

        max_i = np.argmax(summed_dump[:, e_type])
        max_score = summed_dump[max_i, e_type]

        ave_score = np.average(summed_dump[:, e_type])

        evaluations[gen, 0] = min_score
        evaluations[gen, 1] = max_score
        evaluations[gen, 2] = ave_score

    evaluations /= Settings.Simulation.TOTAL_STEP
    xs = np.arange(0, evaluations.shape[0])

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(xs, evaluations[:, 2])
    axis.fill_between(xs, evaluations[:, 0], evaluations[:, 1], color="gray", alpha=0.3)

    fig.savefig(
        os.path.join(workdir, "evaluation_for_each_generation.svg")
    )
    plt.close(fig)
