import os.path

import numpy as np
from matplotlib import pyplot as plt

from lib.utils import get_head_hash, load_parameter, BaseDataCollector

from src.exp.pushing_food_with_pheromone.settings import HyperParameters, TaskGenerator, Task


class DataCollector(BaseDataCollector):
    def __init__(self):
        super().__init__()
        self.episode = int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5)

        self.sight_output = np.zeros((
            self.episode,
            len(HyperParameters.Environment.BOT_POS),
            3
        ))

        self.olfactory_output = np.zeros((
            self.episode,
            len(HyperParameters.Environment.BOT_POS),
            3
        ))

        self.output = np.zeros((
            self.episode,
            len(HyperParameters.Environment.BOT_POS),
            3
        ))

    def get_episode_length(self) -> int:
        return self.episode

    def _record(self, task: Task, time: int, evaluation: float):
        for bi, b in enumerate(task.get_bots()):
            self.sight_output[time, bi, :] = b.debug_data["s1l4"]
            self.olfactory_output[time, bi, :] = b.debug_data["s2l3"]
            self.output[time, bi, :] = b.debug_data["s3l1"]


def collect_data(
        project_directory: str,
):
    collector = DataCollector()

    task_generator = TaskGenerator()
    para = load_parameter(
        dim=task_generator.get_dim(),
        working_directory=project_directory,
        git_hash=get_head_hash(),
        queue_index=-1
    )
    task = task_generator.generate(para, True)

    collector.run(task)

    return collector.sight_output, collector.olfactory_output, collector.output


def generate_color_list(n, alpha):
    cmap = plt.get_cmap('tab20')
    colors = [(*cmap(i / n)[:3], alpha) for i in range(n)]
    return colors


def main():
    project_directory = "../../../../"

    save_file_name = "data(output)"
    if os.path.exists(save_file_name + ".npz"):
        data = np.load(save_file_name + ".npz")
        sight_output = data["sight_output"]
        olfactory_output = data["olfactory_output"]
        output = data["output"]
    else:
        sight_output, olfactory_output, output = collect_data(project_directory)
        np.savez(
            os.path.basename(save_file_name),
            sight_output=sight_output,
            olfactory_output=olfactory_output,
            output=output
        )

    n = output.shape[0]
    x = np.arange(0, n, dtype=np.float32) / n * HyperParameters.Simulator.EPISODE

    for bi in range(len(HyperParameters.Environment.BOT_POS)):
        fig = plt.figure(figsize=(16, 4), dpi=30)

        for i in range(3):
            ax1 = fig.add_subplot(1, 3, i + 1)
            ax2 = ax1.twinx()
            ax1.plot(x, sight_output[:, bi, i], color="Red", label="sight")
            ax1.plot(x, olfactory_output[:, bi, i], color="Green", label="olfactory")
            ax2.plot(x, output[:, bi, i], color="Blue", label="output")

        fig.legend()

        fig.savefig(os.path.join(project_directory, f"outputs_bot{bi}.svg"))


if __name__ == '__main__':
    main()
