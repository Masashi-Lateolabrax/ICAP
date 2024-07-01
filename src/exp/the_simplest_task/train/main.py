from src.exp.the_simplest_task.settings import HyperParameters, NeuralNetwork, TaskGenerator
from lib.optimizer import CMAES, MultiThreadProc
from lib.utils import get_head_hash


def main():
    dim = NeuralNetwork().num_dim()

    cmaes = CMAES(
        dim=dim,
        generation=HyperParameters.Optimization.GENERATION,
        population=HyperParameters.Optimization.POPULATION,
        sigma=HyperParameters.Optimization.SIGMA
    )
    for _ in range(1, 1 + cmaes.get_generation()):
        task_generator = TaskGenerator()
        cmaes.optimize_current_generation(task_generator, MultiThreadProc)

    head_hash = get_head_hash()[0:8]
    cmaes.get_history().save(f"history_{head_hash}.npz")


if __name__ == '__main__':
    main()
