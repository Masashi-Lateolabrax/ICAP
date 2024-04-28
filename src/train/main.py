from src.settings import NeuralNetwork, TaskGenerator
from src.optimizer import CMAES, MultiThreadProc


def main():
    dim = sum([p.numel() for p in NeuralNetwork().parameters() if p.requires_grad])
    cmaes = CMAES(dim, 3, 30, sigma=0.3, split_tasks=4)
    for gen in range(1, 1 + cmaes.get_generation()):
        env_creator = TaskGenerator()
        cmaes.optimize_current_generation(env_creator, MultiThreadProc)
    cmaes.get_history().save("../history.npz")


if __name__ == '__main__':
    main()
