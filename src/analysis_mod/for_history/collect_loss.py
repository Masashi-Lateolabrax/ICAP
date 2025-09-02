import os
import pickle

from framework.prelude import *

import numpy as np

from ..structure.sim_interface import SimulatorForDebugInterface
from ..for_individual.run import run


def collect_loss(
        save_dir: str,
        settings: Settings,
        recorder: IndividualRecorder,
        simulator: type[SimulatorForDebugInterface],
        file_name: str = "loss_history.pkl"
):
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path):
        print(f"Loss history already exists at {file_path}. Skipping collection.")
        return

    train_loss = []
    validation_loss = [[], [], []]

    for rec in recorder:
        rec: Rec = rec
        train_loss.append(rec.best_fitness)

        ind = rec.best_individual
        for seed in [0, 1, 2]:
            ind._generation = seed
            simulation = simulator(settings, ind, render=False)
            run(settings, simulation)
            validation_loss[seed].append(simulation.loss())
        ind._generation = rec.generation

    losses = np.array([
        train_loss,
        validation_loss[0],
        validation_loss[1],
        validation_loss[2]
    ]).T  # shape: (n_generation, 4). losses[:, 0]: train, losses[:, 1-3]: validation with different random seeds.

    with open(file_path, 'wb') as f:
        pickle.dump(losses, f)
