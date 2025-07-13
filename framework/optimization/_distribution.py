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

    def _solve_lp(self, n: int, max_: int = 10) -> np.ndarray:
        from scipy.optimize import linprog

        max_ = [v * max_ for v in self.throughput.values()]
        weight = np.array([1 / v for v in self.throughput.values()])
        num_weight = len(weight)

        c = np.zeros(num_weight + 1)
        c[-1] = 1

        A_ub = np.zeros((n, num_weight + 1))
        b_ub = np.zeros(n)

        A_ub[:, :-1] = np.eye(n) * weight
        A_ub[:, -1] = -1

        A_eq = np.zeros((1, num_weight + 1))
        b_eq = np.array([n])

        A_eq[0, :n] = 1

        bounds = [(0, m) for m in max_] + [(None, None)]

        res = linprog(
            c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )

        return res.x[:num_weight]

    def update(
            self,
            num_ready_individuals: int,
            socket_status: dict[socket.socket, SocketState],
    ) -> None:
        self._mut_init_batch_size_and_throughput(socket_status)

        if not self.throughput:
            return

        newbies = sum(self.batch_size.values())
        num_tasks = num_ready_individuals - newbies
        if not num_tasks <= 0:
            return

        probability = self._solve_lp(num_tasks)
        probability /= num_tasks

        selected_indices = np.random.choice(
            list(self.throughput.keys()), size=num_tasks, p=probability
        )
        for key in selected_indices:
            self.batch_size[key] += 1

    def get_batch_size(self, sock: socket.socket) -> Optional[int]:
        if not sock in self.batch_size:
            logging.warning("If it works as designed, this should never happen")
            return None

        limit = math.ceil(self.throughput[sock] * 10) if sock in self.throughput else 1

        return min(self.batch_size[sock], limit)
