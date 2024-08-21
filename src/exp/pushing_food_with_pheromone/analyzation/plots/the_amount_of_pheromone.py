import os.path

import numpy as np
from matplotlib import pyplot as plt

from lib.utils import get_head_hash, load_parameter, BaseDataCollector

from src.exp.pushing_food_with_pheromone.settings import HyperParameters, TaskGenerator, Task


class DataCollector(BaseDataCollector):
    def __init__(self):
        super().__init__()
        self.episode = int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5)
        self.data = np.zeros(self.episode)

    def get_episode_length(self) -> int:
        return self.episode

    def pre_record(self, task, time: int):
        pass

    def record(self, task: Task, time: int, evaluation: float):
        self.data[time] = np.max(task.mujoco._pheromone.get_all_gas())


def collect_data(
        project_directory: str,
        git_hash: str,
        generation: int,
) -> np.ndarray:
    collector = DataCollector()

    task_generator = TaskGenerator()
    para = load_parameter(
        dim=task_generator.get_dim(),
        working_directory=project_directory,
        git_hash=git_hash,
        queue_index=generation
    )
    task = task_generator.generate(para, True)

    collector.run(task)

    return collector.data


def generate_color_list(n, alpha):
    cmap = plt.get_cmap('tab20')
    colors = [(*cmap(i / n)[:3], alpha) for i in range(n)]
    return colors


def main():
    project_directory = "../../../../"
    git_hash = get_head_hash()
    generation = -1

    save_file_name = "data(pheromone)"
    if os.path.exists(save_file_name + ".npz"):
        data = np.load(save_file_name + ".npz")
    else:
        data = collect_data(project_directory, git_hash, generation)
        np.save(os.path.basename(save_file_name), data)

    x = np.arange(0, data.shape[0], dtype=np.float32) / data.shape[0] * HyperParameters.Simulator.EPISODE

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, data)
    fig.savefig(os.path.join(project_directory, "max_pheromone.svg"))


if __name__ == '__main__':
    main()
