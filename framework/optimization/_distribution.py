import math
import logging
from typing import Optional, Iterable

import numpy as np

from ._connection import Connection


class Distribution:
    def __init__(self):
        self.probabilities: dict[str, Optional[float]] = {}
        self.batch_size: dict[str, int] = {}
        self.performance: dict[str, Optional[float]] = {}

    def add_new_connection(self, conn: Connection) -> None:
        if conn.address in self.performance:
            logging.warning(f"Connection {conn.address} already exists in performance tracking.")
            return

        self.performance[conn.address] = None
        self.batch_size[conn.address] = 1

    def _remove_unhealthy_connection_info(self, connections: Iterable[Connection]) -> None:
        for conn in connections:
            if not conn.is_healthy:
                if conn.address in self.performance:
                    logging.warning(f"Connection {conn.address} is unhealthy, removing from performance tracking.")
                    # Remove unhealthy connection from performance tracking
                    del self.performance[conn.address]
                continue

    def _update_batch_size(self, num_ready_individuals: int) -> None:
        keys = list(self.performance.keys())
        if not keys:
            logging.warning("No connections available for distribution update")
            return

        self.batch_size = {}

        probabilities: dict[str, Optional[float]] = {}
        for k in keys:
            p = self.performance[k]
            if p is not None and p > 0:
                probabilities[k] = p
            else:
                # Set batch size to 1 at least for connections with no performance or zero throughput
                self.batch_size[k] = 1

        population_size = num_ready_individuals - len(self.batch_size)
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

    def register_throughput(self, conn, throughput: float) -> None:
        if self.performance[conn.address] is None:
            self.performance[conn.address] = throughput
        else:
            self.performance[conn.address] = 0.8 * self.performance[conn.address] + 0.2 * throughput

    def update(self, num_ready_individuals: int, connections: Iterable[Connection]) -> None:
        self._remove_unhealthy_connection_info(connections)
        self._update_batch_size(num_ready_individuals)

    def get_batch_size(self, conn: Connection) -> Optional[int]:
        if not conn.is_healthy:
            logging.warning(f"Connection {conn.address} is unhealthy, returning None for batch size.")
            return None
        elif conn.address not in self.batch_size:
            logging.warning(f"Connection {conn.address} not found in batch size tracking, returning None.")
            return None

        batch_size = self.batch_size[conn.address]
        n = math.floor(max(conn.throughput * 10, 1))

        return min(batch_size, n)
