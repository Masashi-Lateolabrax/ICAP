from src.exp.pushing_food.settings import HyperParameters, NeuralNetwork, TaskGenerator
from lib.optimizer import CMAES, MultiThreadProc
from lib.utils import get_head_hash


def main():
    dim = sum([p.numel() for p in NeuralNetwork().parameters() if p.requires_grad])
    cmaes = CMAES(
        dim=dim,
        generation=HyperParameters.Optimization.GENERATION,
        population=HyperParameters.Optimization.POPULATION,
        sigma=HyperParameters.Optimization.SIGMA
    )
    for gen in range(1, 1 + cmaes.get_generation()):
        task_generator = TaskGenerator()
        cmaes.optimize_current_generation(task_generator, MultiThreadProc)

    head_hash = get_head_hash()
    cmaes.get_history().save(f"../history_{head_hash}.npz")


if __name__ == '__main__':
    main()
