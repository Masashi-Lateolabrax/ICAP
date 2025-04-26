import os
import datetime

import numpy as np
import matplotlib.pyplot as plt

import framework

from loss import Loss
from settings import Settings
from logger import Logger
from brain import BrainBuilder


def plot_loss(settings: Settings, dump: framework.Dump, file_path):
    loss = Loss()

    nest_pos = np.array(settings.Nest.POSITION)

    loss_log = np.zeros((len(dump), loss.num_elements()))

    for i in range(len(dump)):
        delta = dump[i]
        food_pos = delta.food_pos
        robot_pos = np.array(list(delta.robot_pos.values()))
        loss_log[i, :] = loss.calc_elements(nest_pos, robot_pos, food_pos)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    loss_lines = []
    for loss, (label, c_name) in zip(loss_log.T, [("distance", "blue")]):
        loss_lines.append(
            ax.plot(loss, label=label, color=c_name)[0]
        )

    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title('Loss')

    fig.legend(handles=loss_lines, loc='lower center')

    fig.savefig(file_path)


def record_in_mp4(settings: Settings, save_dir, logger: Logger, brain_builder: BrainBuilder):
    settings.Robot.ARGMAX_SELECTION = True
    for g in set(list(range(0, len(logger), max(len(logger) // 10, 1))) + [len(logger) - 1]):
        para = logger[g].min_ind
        file_path = os.path.join(save_dir, f"gen{g}.mp4")

        ## Record the simulation.
        dump = framework.entry_points.record(
            settings,
            file_path,
            para,
            brain_builder,
            debug=True
        )

        ## Plot the loss.
        plot_loss(settings, dump, os.path.join(save_dir, f"loss_gen{g}.png"))


def plot_parameter_movements(logger: Logger, file_path, start=0, end=None):
    prev_para = logger[start].min_ind
    movement = []
    for queue in logger[start:end]:
        distance = np.linalg.norm(queue.min_ind - prev_para)
        movement.append(distance)

    xs = np.arange(len(movement))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, movement)
    fig.savefig(file_path)


def test_suboptimal_individuals(
        save_dir: str,
        logger: Logger,
        settings: Settings,
        brain_builder: framework.interfaces.BrainBuilder
):
    settings.Robot.ARGMAX_SELECTION = True
    task_generator = framework.TaskGenerator(settings, brain_builder)

    losses = []
    time = datetime.datetime.now()

    for i in range(len(logger)):
        para = logger[i].min_ind
        task = task_generator.generate(para, debug=True)

        loss = 0
        for t in range(settings.Simulation.TIME_LENGTH):
            loss += task.calc_step()

            if datetime.datetime.now() - time > datetime.timedelta(seconds=1):
                time = datetime.datetime.now()
                print(
                    f"Generation: {i}/{len(logger)}, Progress: {t / settings.Simulation.TIME_LENGTH * 100}%, loss: {loss}"
                )

        losses.append(loss)

    xs = np.arange(len(losses))
    file_path = os.path.join(save_dir, f"test_loss.png")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, losses)
    fig.savefig(file_path)

    return losses
