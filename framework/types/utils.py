import pickle

from .optimization import Individual


class SavedIndividual:
    def __init__(self, generation, avg_fitness, timestamp, individuals: list[Individual]):
        self.generation = generation
        self.avg_fitness = avg_fitness
        self.timestamp = timestamp
        self.individuals = individuals
        self.num_individuals = len(individuals)

        fitnesses = [ind.get_fitness() for ind in individuals]
        self.best_fitness = max(fitnesses)
        self.worst_fitness = min(fitnesses)

    @staticmethod
    def load(path: str) -> 'SavedIndividual':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @property
    def best_individual(self):
        return min(self.individuals, key=lambda ind: ind.get_fitness())
