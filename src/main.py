from viewer import App
from brain import NeuralNetwork
from environment import ECreator

from optimizer.cmaes import base


def entry_point():
    n = 1
    timestep = 0.01
    dim = sum([p.numel() for p in NeuralNetwork().parameters() if p.requires_grad])

    from optimizer import CMAES
    cmaes = CMAES(dim, 1000, 100, sigma=0.3)
    for gen in range(1, 1 + cmaes.get_generation()):
        env_creator = ECreator(n, n, 3, timestep)
        cmaes.optimize_current_generation(gen, env_creator, base.OneThreadProc)
    cmaes.get_history().save("history.npz")
    para = cmaes.get_best_para()

    # history = base.Hist.load("history_4318e5cf.npz")
    # para = history.queues[-1].min_para

    # import numpy as np
    # para = np.random.default_rng().random(dim)

    env_creator = ECreator(n, n, 1, timestep)
    env = env_creator.create(para)

    app = App(500, 500, env)
    app.mainloop()


if __name__ == '__main__':
    entry_point()
