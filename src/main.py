import numpy as np

from viewer import App
from brain import NeuralNetwork
from environment import ECreator


def entry_point():
    n = 1
    dim = sum([p.numel() for p in NeuralNetwork().parameters() if p.requires_grad])
    env_creator = ECreator(n, n)

    # from optimizer import CMAES
    # cmaes = CMAES(dim, 30, 50, max_thread=14)
    # cmaes.optimize(env_creator)
    # cmaes.get_history().save("history.npz")

    rng = np.random.default_rng()

    # env = env_creator.create(cmaes.get_best_para())
    env = env_creator.create(rng.random(dim))
    app = App(500, 500, env)
    app.mainloop()


if __name__ == '__main__':
    entry_point()
