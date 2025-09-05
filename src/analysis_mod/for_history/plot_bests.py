import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


def plot_fitness(save_dir: str, filepath: str):
    with open(os.path.join(save_dir, "loss_history.pkl"), 'rb') as f:
        # shape: (n_generation, 2). losses[:, 0]: train, losses[:, 1-3]: validation with different random seeds.
        losses = pickle.load(f)

    train_fitness = losses[:, 0]
    val_fitness = np.mean(losses[:, 1:], axis=1)

    gen_best_in_best = np.argmin(train_fitness)
    gen_best_in_val = np.argmin(val_fitness)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(train_fitness, label="Train Fitness", color='blue')
    ax.plot(val_fitness, label="Validation Fitness", color='orange')

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Fitness Over Generations\n(Train: {gen_best_in_best}, Val: {gen_best_in_val})")
    ax.legend()

    fig.savefig(filepath)
