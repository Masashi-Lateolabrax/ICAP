import os

from libs.optimizer import Individual

import framework
from framework.analysis import plot_parameter_movements, LabelAndColor

from loss import Loss
from settings import Settings
from brain import BrainBuilder
from logger import Logger


def plot_loss(file_path: str, settings: Settings, ind: Individual):
    framework.analysis.plot_loss(
        file_path,
        settings,
        ind,
        Loss(),
        labels={
            0: LabelAndColor(label="robot and food", color="blue"),
            1: LabelAndColor(label="food and nest", color="orange"),
        }
    )


def record_in_mp4(save_dir: str, settings: Settings, logger: Logger, brain_builder: BrainBuilder):
    os.makedirs(save_dir, exist_ok=True)

    settings.Robot.ARGMAX_SELECTION = True
    task_generator = framework.TaskGenerator(settings, brain_builder)

    with open(os.path.join(save_dir, "robot_food_pos.log.txt"), "w") as f:
        for i, rp in enumerate(task_generator.robot_positions):
            f.write(f"Robot{i}: {repr(rp[0])}, {repr(rp[1])}, {repr(rp[2])}\n")
        for i, fp in enumerate(task_generator.food_positions):
            f.write(f"Food{i}: {repr(fp[0])}, {repr(fp[1])}\n")

    for g in set(list(range(0, len(logger), max(len(logger) // 10, 1))) + [len(logger) - 1]):
        ind = logger[g].max_ind
        task = task_generator.generate(ind, debug=True)

        ## Record the simulation.
        ind.dump = framework.entry_points.record(
            settings,
            os.path.join(save_dir, f"gen{g}.mp4"),
            task,
        )

        ## Plot the loss.
        framework.analysis.plot_loss(
            os.path.join(save_dir, f"loss_gen{g}.png"),
            settings,
            ind,
            Loss(),
            labels={
                0: LabelAndColor(label="robot and food", color="blue"),
                1: LabelAndColor(label="food and nest", color="orange"),
            }
        )


def test_suboptimal_individuals(
        file_path: str,
        settings: Settings,
        logger: Logger,
        brain_builder: BrainBuilder,
):
    settings.Robot.ARGMAX_SELECTION = True

    task_generator = framework.TaskGenerator(settings, brain_builder)

    return framework.analysis.test_suboptimal_individuals(
        file_path,
        Settings(),
        logger,
        task_generator
    )


def plot_max_of_parameter(
        file_path: str,
        logger: Logger,
        start: int = 0,
        end: int = None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    end = end if end is not None else len(logger)
    individuals = [logger[i].max_ind for i in range(start, end)]

    max_values = [np.max(ind) for ind in individuals]
    min_values = [np.min(ind) for ind in individuals]
    ave_values = [np.mean(ind) for ind in individuals]
    std_values = [np.std(ind) for ind in individuals]
    generation = np.arange(len(max_values))

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(generation, max_values, label="max of parameter")
    axis.plot(generation, min_values, label="min of parameter")
    axis.plot(generation, ave_values, label="ave of parameter")
    axis.fill_between(
        generation,
        np.array(ave_values) - np.array(std_values),
        np.array(ave_values) + np.array(std_values),
        alpha=0.2,
        label="std of parameter"
    )
    plt.savefig(file_path)


def plot_parameter_heatmap(
        file_path: str,
        logger,
        start: int = 0,
        end: int = None
):
    import numpy as np
    import matplotlib.pyplot as plt

    end = end if end is not None else len(logger)
    parameter_matrix = np.array([logger[i].max_ind for i in range(start, end)])

    plt.figure(figsize=(10, 6))
    plt.imshow(parameter_matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Parameter Value')
    plt.xlabel('Generation')
    plt.ylabel('Parameter Index')
    plt.title('Parameter Heatmap Over Generations')
    plt.savefig(file_path)
    plt.close()


def plot_loss_during_train(
        file_path: str,
        logger: Logger,
        start: int = 0,
        end: int = None
):
    from matplotlib import pyplot as plt

    worst_values = []
    best_values = []
    average_values = []

    for queue in logger[start:end]:
        worst = queue.min_ind
        best = queue.max_ind

        worst_values.append(worst.fitness.values[0])
        best_values.append(best.fitness.values[0])
        average_values.append(queue.scores_avg)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(worst_values, label="worst")
    axis.plot(best_values, label="best")
    axis.plot(average_values, label="average")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.savefig(file_path)
