import logging
import math
import socket
from typing import Optional

import numpy as np

from ..prelude import *


class Distribution:
    def __init__(self):
        self.throughput: dict[socket.socket, float] = {}
        self.batch_size: dict[socket.socket, int] = {}

    def _mut_init_batch_size_and_throughput(self, socket_status: dict[socket.socket, SocketState]) -> None:
        self.batch_size = {}
        self.throughput = {}
        for sock, state in socket_status.items():
            if state.throughput > 0:
                self.batch_size[sock] = 0  # This value will be counted up later (LINE 49)
                self.throughput[sock] = state.throughput
            else:
                self.batch_size[sock] = 1

    def update(
            self,
            num_ready_individuals: int,
            socket_status: dict[socket.socket, SocketState],
    ) -> None:
        self._mut_init_batch_size_and_throughput(socket_status)

        if not self.throughput:
            return

        probability = {}
        for key, throughput in self.throughput.items():
            probability[key] = throughput
        total = sum(probability.values())
        probability = {key: throughput / total for key, throughput in probability.items()}

        newbies = sum(self.batch_size.values())
        num_tasks = num_ready_individuals - newbies
        if not probability or num_tasks <= 0:
            return

        selected_indices = np.random.choice(
            list(probability.keys()), size=num_tasks, p=list(probability.values())
        )
        for key in selected_indices:
            self.batch_size[key] += 1

    def get_batch_size(self, sock: socket.socket) -> Optional[int]:
        if not sock in self.batch_size:
            logging.warning("If it works as designed, this should never happen")
            return None

        limit = math.ceil(self.throughput[sock] * 10) if sock in self.throughput else 1

        return min(self.batch_size[sock], limit)
