import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt

from lib.utils import get_head_hash, load_parameter

from src.exp.pushing_food_with_pheromone.settings import HyperParameters, NeuralNetwork


def collect_data(
        project_directory: str,
        max_value: int,
        min_value: int,
        precision: int
) -> tuple[np.ndarray, np.ndarray]:
    brain = NeuralNetwork()
    para = load_parameter(
        dim=brain.num_dim(),
        working_directory=project_directory,
        git_hash=get_head_hash(),
        queue_index=-1
    )
    brain.load_para(para)

    brain.requires_grad_(False)

    x = np.array(
        list(map(lambda p: p / precision, range(min_value * precision, int(max_value * precision))))
    )
    input_buf = torch.zeros(1)
    data = np.zeros((x.shape[0], 3))
    for t in range(x.shape[0]):
        input_buf[0] = float(x[t])
        value = brain.sequence2.forward(input_buf)
        data[t, :] = value.detach().numpy()

    return x, data


def main():
    project_directory = "../../../../"

    save_file_name = "data(olfactory)"
    if os.path.exists(save_file_name + ".npz"):
        raw = np.load(save_file_name + ".npz")
        x = raw["x"]
        data = raw["data"]
    else:
        x, data = collect_data(
            project_directory=project_directory,
            max_value=int(HyperParameters.Pheromone.SaturatedVapor),
            min_value=-1,
            precision=100
        )
        np.savez(
            os.path.basename(save_file_name),
            x=x,
            data=data,
        )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, data[:, 0], color="Red")
    ax.plot(x, data[:, 1], color="Green")
    ax.plot(x, data[:, 2], color="Blue")
    fig.savefig(os.path.join(project_directory, f"olfactory.svg"))


if __name__ == '__main__':
    main()
