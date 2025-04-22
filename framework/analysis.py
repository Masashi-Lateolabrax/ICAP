import dataclasses
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt

from libs.optimizer import EachGenLogger, Individual

from .optimization import Loss
from .settings import Settings
from .simulator.objects import BrainBuilder
from .task_generator import TaskGenerator

from . import entry_points


@dataclasses.dataclass
class LabelAndColor:
    label: str = None
    color: str = None


def plot_loss(
        file_path: str,
        settings: Settings,
        ind: Individual,
        loss: Loss,
        labels: dict[int, LabelAndColor] = None
):
    if labels is None:
        labels = {}
    dump = ind.dump
    score = ind.fitness.values[0]

    if np.isinf(score):
        print(f"Individual {ind} is invalid, skipping...")

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
    for i, loss in enumerate(loss_log.T):
        label = labels.get(i, LabelAndColor())
        loss_lines.append(
            ax.plot(loss, label=label.label, color=label.color)[0]
        )

    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title('Loss')

    fig.legend(handles=loss_lines, loc='lower center')

    fig.savefig(file_path)


def record_in_mp4(
        save_dir: str,
        settings: Settings,
        logger: EachGenLogger,
        loss: Loss,
        brain_builder: BrainBuilder,
        labels: dict[int, LabelAndColor] = None
):
    os.makedirs(save_dir, exist_ok=True)

    settings.Robot.ARGMAX_SELECTION = True
    task_generator = TaskGenerator(settings, brain_builder)

    for g in set(list(range(0, len(logger), max(len(logger) // 10, 1))) + [len(logger) - 1]):
        ind = logger[g].min_ind
        task = task_generator.generate(ind, debug=True)

        ## Record the simulation.
        ind.dump = entry_points.record(
            settings,
            os.path.join(save_dir, f"gen{g}.mp4"),
            task,
        )

        ## Plot the loss.
        plot_loss(
            os.path.join(save_dir, f"loss_gen{g}.png"),
            settings,
            ind,
            loss,
            labels
        )


def plot_parameter_movements(file_path: str, logger: EachGenLogger, start=0, end=None):
    dim = logger[start].min_ind.shape[0]

    subs = np.array([logger[i + 1].min_ind - logger[i].min_ind for i in range(start, (end or len(logger)) - 1)])
    movement = np.linalg.norm(subs, axis=1)

    # direction[i] += a[i,j] * b[i,j]
    direction = np.einsum(
        "ij,ij->i",
        subs[:-1, :] / movement[:-1, None],
        subs[1:, :] / movement[1:, None]
    )

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    ax1.plot(np.arange(movement.shape[0]) + 0.5, movement / dim, label="movement", color="blue")
    ax2.plot(np.arange(direction.shape[0]) + 1, direction, label="direction", color="orange")

    fig.legend(loc="lower center")

    fig.savefig(file_path)


def test_suboptimal_individuals(
        file_path: str,
        settings: Settings,
        logger: EachGenLogger,
        task_generator: TaskGenerator,
):
    settings.Robot.ARGMAX_SELECTION = True

    losses = []
    time = datetime.datetime.now()

    for i in range(len(logger)):
        para = logger[i].min_ind
        task = task_generator.generate(para, debug=True)

        loss = 0
        for t in range(settings.Simulation.TIME_LENGTH):
            loss += task.calc_step()
            if np.isinf(loss):
                print(f"generation {i} is invalid, skipping...")
                loss = float("nan")
                break

            if datetime.datetime.now() - time > datetime.timedelta(seconds=1):
                time = datetime.datetime.now()
                print(
                    f"Generation: {i}/{len(logger)}, Progress: {t / settings.Simulation.TIME_LENGTH * 100}%, loss: {loss}"
                )

        losses.append(loss)

    xs = np.arange(len(losses))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, losses)
    fig.savefig(file_path)

    return losses
