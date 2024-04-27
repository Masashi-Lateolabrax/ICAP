from viewer import App
from brain import NeuralNetwork
from environment import ECreator

from optimizer.cmaes import base


def entry_point():
    n = 1
    dim = sum([p.numel() for p in NeuralNetwork().parameters() if p.requires_grad])

    from optimizer import CMAES
    cmaes = CMAES(dim, 1000, 100, sigma=0.03)
    for gen in range(1, 1 + cmaes.get_generation()):
        env_creator = ECreator(n, n)
        cmaes.optimize_current_generation(gen, env_creator, base.OneThreadProc)
    cmaes.get_history().save("history.npz")

    env_creator = ECreator(n, n)
    env = env_creator.create(cmaes.get_best_para())

    # rng = np.random.default_rng()
    # env = env_creator.create(rng.random(dim))

    app = App(500, 500, env)
    app.mainloop()


if __name__ == '__main__':
    entry_point()
