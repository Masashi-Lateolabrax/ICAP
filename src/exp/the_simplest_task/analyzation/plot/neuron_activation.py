import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from lib.utils import BaseDataCollector, load_parameter, get_head_hash

from src.exp.the_simplest_task.settings import HyperParameters, TaskGenerator, Task


class DataCollector(BaseDataCollector):
    def __init__(self, investigator_name: str):
        super().__init__()
        self.investigator_name = investigator_name
        self.episode = int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5)
        self.data = np.zeros((1, 1))

    def get_episode_length(self) -> int:
        return self.episode

    def pre_record(self, task, time: int):
        pass

    def record(self, task: Task, time: int, evaluation: float):
        debug_data = task.get_bot().brain.debugger.get_buf()[self.investigator_name]
        if self.data.shape[1] != debug_data.shape[0]:
            self.data = np.zeros((self.episode, debug_data.shape[0]))
        self.data[time, :] = debug_data


def collect_data(project_directory: str, git_hash: str, investigator_name: str) -> np.ndarray:
    collector = DataCollector(investigator_name)

    task_generator = TaskGenerator()
    para = load_parameter(
        dim=task_generator.get_dim(),
        working_directory=project_directory,
        git_hash=git_hash,
        queue_index=-1
    )
    task = task_generator.generate(para, True)

    collector.run(task)

    return collector.data


def plot_animation(project_directory: str, data: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def frame():
        time = datetime.now()
        length = data.shape[0]

        for i in range(length):
            if (datetime.now() - time).seconds > 1:
                time = datetime.now()
                print(f"{i}/{length}")

            yield i, data[i]

    def update(f):
        i, activation = f

        ax.clear()
        ax.set_title(f"Frame {i}")

        activation = np.expand_dims(activation, 0)
        ax.matshow(activation)

        return fig.artists

    ani = animation.FuncAnimation(fig, update, frames=frame, blit=False, cache_frame_data=False)
    ani.save(os.path.join(project_directory, 'animation.mp4'))


def main():
    project_directory = "../../../../"
    investigator_name = "l0"
    git_hash = get_head_hash()

    save_file_name = "data(activation)"
    if os.path.exists(save_file_name + ".npz"):
        data = np.load(save_file_name + ".npz")
    else:
        data = collect_data(project_directory, git_hash, investigator_name)
        np.save(os.path.basename(save_file_name), data)

    plot_animation(project_directory, data)


if __name__ == '__main__':
    main()
