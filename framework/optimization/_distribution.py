import numpy as np

import logging
from typing import Optional

from ._connection import Connection
from ._cmaes import CMAES


class Distribution:
    def __init__(self, cmaes: CMAES):
        self.population_size = cmaes.population_size
        self.batch_size: dict[str, int] = {}
        self.performance: dict[str, Optional[float]] = {}

    def register_performance(self, conn: Connection) -> None:
        if not conn.is_healthy:
            logging.warning(f"Connection {conn.address} is unhealthy, skipping performance registration.")
            return

        throughput = self.performance[conn.address] = conn.throughput

        logging.debug(f"Registered performance for {conn.address}: {throughput:.2f} ind/sec")

    def update(self) -> None:
        self.batch_size = {}
        keys = list(self.performance.keys())

        if not keys:
            logging.warning("No connections available for distribution update")
            return

        probabilities: dict[str, Optional[float]] = {}
        for k in keys:
            p = self.performance[k]
            if p is not None and p > 0:
                probabilities[k] = p
            else:
                # Set batch size to 1 at least for connections with no performance or zero throughput
                self.batch_size[k] = 1

        population_size = self.population_size - len(self.batch_size)
        if not probabilities or population_size <= 0:
            return

        total = sum(probabilities.values())
        for key in probabilities:
            probabilities[key] = probabilities[key] / total

        num_connections = len(probabilities)

        # Each values of 'probabilities' is greater than 0 absolutely. Because we filtered out
        # connections with zero or None performance above.
        selected_indices = np.random.choice(
            np.arange(num_connections), size=population_size, p=list(probabilities.values())
        )
        task_distribution = np.bincount(selected_indices, minlength=num_connections)

        for key, num in zip(probabilities.keys(), task_distribution):
            self.batch_size[key] = num

        # We need to reset performance because of dropping disconnected connections.
        # Connection performance values are registered when Server receives fitness values.
        # So, this reset is no problem.
        self.performance = {}

    def get_batch_size(self, conn: Connection) -> Optional[int]:
        if not conn.is_healthy:
            logging.warning(f"Connection {conn.address} is unhealthy, returning None for batch size.")
            return None

        return self.batch_size.get(conn.address, None)
