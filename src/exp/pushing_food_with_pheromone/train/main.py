from src.exp.pushing_food_with_pheromone.settings import HyperParameters, NeuralNetwork, TaskGenerator
from lib.optimizer import CMAES, MultiThreadProc
from lib.utils import get_head_hash


def main():
    print("PUSHING FOOD WITH PHEROMONE")

    dim = NeuralNetwork().num_dim()

    cmaes = CMAES(
        dim=dim,
        generation=HyperParameters.Optimization.GENERATION,
        population=HyperParameters.Optimization.POPULATION,
        sigma=HyperParameters.Optimization.SIGMA,
        mu=HyperParameters.Optimization.MU
    )
    for gen in range(1, 1 + cmaes.get_generation()):
        task_generator = TaskGenerator(False)
        cmaes.optimize_current_generation(task_generator, MultiThreadProc)

    head_hash = get_head_hash()[0:8]
    cmaes.get_history().save(f"history_{head_hash}.npz")


if __name__ == '__main__':
    main()
