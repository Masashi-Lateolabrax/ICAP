import os.path

import numpy
from matplotlib import pyplot as plt
from environments.collect_feed import EnvCreator, DumpData, RobotBrain
from studyLib.optimizer import Hist
from main import set_env_creator


def collect_data(para, env_creator: EnvCreator) -> DumpData:
    env_creator.dump_data = DumpData()
    env = env_creator.create(para)
    env.calc()
    return env_creator.dump_data


def plot_inputted_pheromone(dump_data: DumpData, ax: plt.Axes, pheromone_index, robot_id, color):
    data = numpy.zeros(int(len(dump_data.queue) / dump_data.num_robots()))
    index = 0
    for q in dump_data.queue:
        for rq in q.robot:
            if rq.robot_id != robot_id:
                continue
            data[index] = rq.inputted_pheromone[pheromone_index]
            index += 1
            break
    x = numpy.arange(0, len(data))
    ax.plot(x, data, color=color)


def plot_pheromone_secretion(dump_data: DumpData, ax: plt.Axes, color):
    data = numpy.zeros((len(dump_data.queue), dump_data.num_robots()))
    for i, q in enumerate(dump_data.queue):
        for rq in q.robot:
            data[i, rq.robot_id] = rq.output[2]
    x = numpy.arange(0, len(data))
    for ri in range(dump_data.num_robots()):
        ax.plot(x, data[:, ri], color=color[ri])


def plot_distance_to_nest(dump_data: DumpData, ax: plt.Axes, color):
    data = numpy.zeros((len(dump_data.queue), dump_data.num_robots()))
    for i, q in enumerate(dump_data.queue):
        for rq in q.robot:
            data[i, rq.robot_id] = rq.inputted_nest[2]
    x = numpy.arange(0, len(data))
    for ri in range(dump_data.num_robots()):
        ax.plot(x, data[:, ri], color=color[ri])


def plot_amount_of_pheromone_in_field(dump_data: DumpData, ax: plt.Axes, pheromone_id: int):
    data = numpy.zeros((len(dump_data.queue), 2))
    for i, q in enumerate(dump_data.queue):
        data[i, 0] = i
        data[i, 1] = q.max_pheromone[pheromone_id]
    ax.plot(data[:, 0], data[:, 1])


def plot_pattern_vs_time(dump_data: DumpData, ax: plt.Axes, robot_id: int):
    data = numpy.zeros((len(dump_data.queue), 2))
    for i, q in enumerate(dump_data.queue):
        data[i, 0] = i
        # data[i, 1] = q.max_pheromone[pheromone_id]
    ax.plot(data[:, 0], data[:, 1])


def plot_pheromone_response(para, ax: plt.Axes, pattern, precision=1000):
    brain = RobotBrain(para)
    decision_part = brain.get_decision_parts()
    input_dimension = decision_part.num_input()
    sv = 10.0

    input_buf = numpy.zeros(input_dimension)
    input_buf[pattern] = 1.0

    data = numpy.zeros((precision, 4))
    for i in range(precision):
        input_buf[input_dimension - 1] = sv * i / precision
        decision = decision_part.calc(input_buf)
        data[i, 0] = input_buf[input_dimension - 1]
        data[i, 1:] = decision

    for i, c in enumerate([(1, 0, 0), (0, 0, 1), (0, 1, 0)]):
        ax.plot(data[:, 0], data[:, i + 1], color=c)


def plot_loss(ax_min: plt.Axes, ax_max: plt.Axes, hist: Hist):
    data = numpy.zeros((len(hist.queues), 4))
    for i, q in enumerate(hist.queues):
        data[i, :] = [i, q.min_score, q.max_score, q.scores_avg]

    ax_min.plot(data[:, 0], data[:, 1], c=(0, 1, 0, 1))
    ax_min.plot(data[:, 0], data[:, 3], c=(0, 0, 1, 1))
    ax_max.plot(data[:, 0], data[:, 2], c=(1, 0, 0, 1))


def plot_l2(ax: plt.Axes, hist: Hist):
    data = numpy.zeros((len(hist.queues), 2))
    for i, q in enumerate(hist.queues):
        data[i, :] = [i, numpy.linalg.norm(q.min_para)]
    ax.plot(data[:, 0], data[:, 1], c=(0, 0.8, 0, 1))


def plot_diff_l2(ax: plt.Axes, hist: Hist):
    data = numpy.zeros((len(hist.queues) - 1, 3))

    for i, (prev_q, next_q) in enumerate(zip(hist.queues[:-1], hist.queues[1:])):
        data[i, 0] = i + 0.5
        data[i, 1] = numpy.linalg.norm(next_q.min_para - prev_q.min_para, ord=2)
        data[i, 2] = numpy.linalg.norm(next_q.max_para - prev_q.max_para, ord=2)

    ax.plot(data[:, 0], data[:, 1], c=(0, 0.8, 0, 1))


def plot_sigma(ax: plt.Axes, hist: Hist):
    data = numpy.zeros((len(hist.queues), 2))
    for i, q in enumerate(hist.queues):
        data[i, :] = [i, q.sigma]
    ax.plot(data[:, 0], data[:, 1], c=(0, 0.8, 0, 1))


if __name__ == '__main__':
    def main():
        numpy.random.seed(230602)

        env_creator = EnvCreator()
        set_env_creator(env_creator)
        hist = Hist.load("history.npz")

        if not os.path.isfile("dump_data"):
            dump_data = collect_data(hist.get_min().min_para, env_creator)
            dump_data.save("dump_data")

        dump_data = DumpData.load("dump_data")

        colors = [
            "#FF0000",
            "#00FF00",
            "#0000FF",
            "#FFA500",
            "#FFFF00",
            "#800080",
            "#FFC0CB",
            "#00FFFF",
            "#FF4500",
            "#ADFF2F"
        ]

        # fig = plt.figure(figsize=(12, 4), dpi=100)
        # fig.subplots_adjust(
        #     right=0.99,
        #     left=0.03,
        #     top=0.95,
        #     bottom=0.07,
        # )
        # ax: plt.Axes = fig.add_subplot(1, 1, 1)
        # plot_pheromone_secretion(dump_data, ax, colors)
        # # fig.show()

        # fig = plt.figure(figsize=(12, 4), dpi=100)
        # fig.subplots_adjust(
        #     right=None,
        #     left=None,
        #     top=0.99,
        #     bottom=0.02,
        # )
        # ax: plt.Axes = fig.add_subplot(1, 1, 1)
        # plot_distance_to_nest(dump_data, ax, colors)
        # fig.show()

        # fig: plt.Figure = plt.figure(figsize=(6, 25), dpi=100)
        # fig.subplots_adjust(
        #     right=None,
        #     left=None,
        #     top=0.99,
        #     bottom=0.02,
        # )
        # for pi in range(0, 1):
        #     for ri, color in enumerate(colors):
        #         ax: plt.Axes = fig.add_subplot(10, 1, ri + 1)
        #         plot_inputted_pheromone(dump_data, ax, pi, ri, color)
        # fig.show()

        # fig: plt.Figure = plt.figure(figsize=(12, 4), dpi=100)
        # fig.subplots_adjust(
        #     right=None,
        #     left=None,
        #     top=0.99,
        #     bottom=0.1,
        # )
        # ax: plt.Axes = fig.add_subplot(1, 1, 1)
        # plot_amount_of_pheromone_in_field(dump_data, ax, 0)
        # fig.show()

        # plot_pattern_vs_time(dump_data, ax, ri)

        fig: plt.Figure = plt.figure(figsize=(6, 18), dpi=100)
        fig.subplots_adjust(
            right=None,
            left=None,
            top=0.99,
            bottom=0.02,
        )
        for i in range(5):
            ax: plt.Axes = fig.add_subplot(10, 1, i + 1)
            ax.set_xlim(0, 3)
            plot_pheromone_response(hist.get_min().min_para, ax, i)
        fig.show()

        # fig: plt.Figure = plt.figure(figsize=(6, 18), dpi=100)
        # fig.subplots_adjust(
        #     right=None,
        #     left=None,
        #     top=0.99,
        #     bottom=0.02,
        # )
        #
        # ax_max: plt.Axes = fig.add_subplot(5, 1, 1)
        # ax_min: plt.Axes = fig.add_subplot(5, 1, 2)
        # plot_loss(ax_min, ax_max, hist)
        #
        # ax: plt.Axes = fig.add_subplot(5, 1, 3)
        # plot_l2(ax, hist)
        #
        # ax: plt.Axes = fig.add_subplot(5, 1, 4)
        # plot_diff_l2(ax, hist)
        #
        # ax: plt.Axes = fig.add_subplot(5, 1, 5)
        # plot_sigma(ax, hist)
        #
        # fig.show()


    main()
