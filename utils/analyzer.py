import numpy
from matplotlib import pyplot as plt
from environments.collect_feed_without_obstacle import EnvCreator
from main import set_env_creator


class DumpData:
    class Queue:
        def __init__(
                self,
                input_values: numpy.ndarray,
                cortex_values: numpy.ndarray,
                output_values: numpy.ndarray
        ):
            self.input_values = input_values.copy()
            self.cortex_values = cortex_values.copy()
            self.output_values = output_values.copy()

    def __init__(self):
        self._queue: list[list[DumpData.Queue]] = []

    def dump(
            self,
            robot_index: int,
            input_values: numpy.ndarray,
            cortex_values: numpy.ndarray,
            output_values: numpy.ndarray
    ):
        while len(self._queue) <= robot_index:
            self._queue.append([])
        self._queue[robot_index].append(
            DumpData.Queue(input_values, cortex_values, output_values)
        )

    def compile(self) -> numpy.ndarray:
        num = len(self._queue)
        time = len(self._queue[0])
        dim = len(self._queue[0][0].input_values)
        dim += len(self._queue[0][0].cortex_values) + len(self._queue[0][0].output_values)

        result = numpy.zeros((time, num, dim))
        for i, r in enumerate(self._queue):
            for t, q in enumerate(r):
                result[t][i] = numpy.concatenate([q.input_values, q.cortex_values, q.output_values])

        return result


def create_dump(para, file_path: str):
    env_creator = EnvCreator()
    set_env_creator(env_creator)

    dump_data = DumpData()

    # width: int = 500
    # height: int = 700
    # scale: int = 1
    # window = miscellaneous.Window(width * scale, height * scale)
    # camera = wrap_mjc.Camera((0, 600, 0), 2500, 90, 90)
    # env = env_creator.create_mujoco_env(para, window, camera)

    env = env_creator.create(para)
    for t in range(0, env_creator.timestep):
        env.calc_step()
        env.render()

        for i, b in enumerate(env.robots):
            input_values = b.brain.get_input()
            cortex_values = b.brain.get_cortexs_output()
            output_values = b.brain.get_output()
            dump_data.dump(i, input_values, cortex_values, output_values)

        print(f"PROGRESS : {100.0 * t / env.timestep}%")

    compiled_dump_data = dump_data.compile()
    numpy.save("dump.log", compiled_dump_data)


def _analyze_cortex_output(axis1: plt.Axes, axis2: plt.Axes, dump_data, robot_index, i):
    x = numpy.arange(0, len(dump_data)) / 30
    axis1.plot(x, dump_data[:, robot_index, i], color=(0, 0, 1, 0.8))  # 嗅覚野
    axis1.plot(x, dump_data[:, robot_index, i + 3], color=(1, 0, 0, 0.8))  # 視覚野
    axis2.plot(x, dump_data[:, robot_index, i + 6], color=(0, 1, 0, 0.8))  # 出力


def plot_cortex_output_for_left_wheel(axis1: plt.Axes, axis2: plt.Axes, dump_data, robot_index):
    axis1.set_ylim(-9.4, 9.4)
    axis2.set_ylim(-1.1, 1.1)
    _analyze_cortex_output(axis1, axis2, dump_data, robot_index, 7)


def plot_cortex_output_for_right_wheel(axis1: plt.Axes, axis2: plt.Axes, dump_data, robot_index):
    axis1.set_ylim(-9.4, 9.4)
    axis2.set_ylim(-1.1, 1.1)
    _analyze_cortex_output(axis1, axis2, dump_data, robot_index, 8)


def plot_cortex_output_for_pheromone(axis1: plt.Axes, axis2: plt.Axes, dump_data, robot_index):
    axis1.set_ylim(-9.4, 9.4)
    axis2.set_ylim(-1.1, 1.1)
    _analyze_cortex_output(axis1, axis2, dump_data, robot_index, 9)


def plot_olfactory_cortex(axis: plt.Axes, para):
    env_creator = EnvCreator()
    set_env_creator(env_creator)
    env = env_creator.create(para)

    olfactory_cortex = env.robots[4].brain.get_olfactory_cortex()

    limit = 10
    precision = 0.1
    data = numpy.zeros((int(limit / precision), 3))
    for p in range(0, int(limit / precision)):
        output = olfactory_cortex.calc([0, 0, 0, 0, 0, 0, p * precision])
        data[p] = output.copy()

    x = numpy.arange(0, int(limit / precision)) * precision
    axis.plot(x, data[:, 0], color=(1, 0, 0, 0.7))
    axis.plot(x, data[:, 1], color=(0, 0, 1, 0.7))
    axis.plot(x, data[:, 2], color=(0, 1, 0, 0.7))


def plot_olfactory_cortex_input(axis: plt.Axes, dump_data: numpy.ndarray, select=None):
    extract = numpy.zeros((len(dump_data), 9))

    color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

    x = numpy.arange(0, len(dump_data)) / 30
    for ri in range(0, 9):
        if select is not None and ri not in select:
            continue
        extract[:, ri] = dump_data[:, ri, 6]
        axis.plot(x, dump_data[:, ri, 6], color=color[ri])

    # numpy.savetxt("olfactory_cortex_input_log.csv", extract, delimiter=',', fmt='%.18e')


if __name__ == "__main__":
    def main():
        para = numpy.load("best_para.npy")

        # create_dump(para, "dump.log")
        dump_data = numpy.load("dump.log.npy")

        # for i, ids in enumerate([[6, 7, 0], [3, 4, 5, 8], [1, 2], [1, 2, 3, 4, 5, 8], None]):  # よくわからん，押し係，止め係
        #     figure: plt.Figure = plt.figure(figsize=(8, 6), dpi=300)
        #     axis1: plt.Axes = figure.add_subplot(1, 1, 1)
        #     axis1.set_yticks([0, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        #     axis1.set_ylim(0, 10)
        #     plot_olfactory_cortex_input(axis1, dump_data, ids)
        #     figure.show()
        #     figure.savefig(f"olfactory_input({i}).png")

        # figure: plt.Figure = plt.figure(figsize=(6, 2), dpi=100)
        # axis1: plt.Axes = figure.add_subplot(1, 1, 1)
        # axis2: plt.Axes = axis1.twinx()
        # plot_cortex_output_for_left_wheel(axis1, axis2, dump_data, 4)
        # figure.show()
        # figure.savefig("left_wheel.png")

        # figure: plt.Figure = plt.figure(figsize=(6, 2), dpi=100)
        # axis1: plt.Axes = figure.add_subplot(1, 1, 1)
        # axis2: plt.Axes = axis1.twinx()
        # plot_cortex_output_for_right_wheel(axis1, axis2, dump_data, 5)
        # figure.show()
        # figure.savefig("right_wheel.png")

        # figure: plt.Figure = plt.figure(figsize=(6, 6), dpi=100)
        # axis1: plt.Axes = figure.add_subplot(3, 1, 1)
        # axis2: plt.Axes = axis1.twinx()
        # plot_cortex_output_for_left_wheel(axis1, axis2, dump_data, 4)
        # axis1: plt.Axes = figure.add_subplot(3, 1, 2)
        # axis2: plt.Axes = axis1.twinx()
        # plot_cortex_output_for_right_wheel(axis1, axis2, dump_data, 4)
        # axis1: plt.Axes = figure.add_subplot(3, 1, 3)
        # axis2: plt.Axes = axis1.twinx()
        # plot_cortex_output_for_pheromone(axis1, axis2, dump_data, 4)
        # figure.show()
        # figure.savefig("right_wheel.png")

        figure: plt.Figure = plt.figure(figsize=(6, 4), dpi=120)
        axis1: plt.Axes = figure.add_subplot(1, 1, 1)
        axis1.set_xticks([0.7, 2, 4, 6, 8, 10])
        axis1.set_xlim(0.7, 10)
        plot_olfactory_cortex(axis1, para)
        figure.show()
        # figure.savefig("olfactory_cortex.png")


    main()
