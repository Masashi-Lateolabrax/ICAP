import uuid
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


class Performance:
    def __init__(self):
        self.performance_time: dict[int, float] = {}

    def register(self, task_count: int, processing_time: float, alpha: float = 0.2):
        current = self.performance_time.get(task_count, 0.0)
        self.performance_time[task_count] = (1 - alpha) * current + alpha * processing_time

    def get(self, task_count: int) -> float:
        if task_count <= 0:
            return 0.0
        return self.performance_time.get(task_count, float("nan"))


class LoadBalancer:
    def __init__(self):
        self.performance_table: dict[uuid.UUID, Performance] = {}
        self._buf_table: dict[uuid.UUID, int] = {}

    def remove(self, id_: uuid.UUID):
        if id_ in self.performance_table:
            del self.performance_table[id_]

    def register_performance(self, id_: uuid.UUID, task_count: int, time: float):
        if id_ not in self.performance_table:
            self.performance_table[id_] = Performance()
        self.performance_table[id_].register(task_count, time)

    def convert_num_tasks_to_time(self, balance: dict[uuid.UUID, int]) -> dict[uuid.UUID, float]:
        result = {}
        for id_, task_count in balance.items():
            if id_ not in self.performance_table:
                result[id_] = float("nan")
                continue
            perf = self.performance_table[id_]
            result[id_] = perf.get(task_count)
        return result

    def calc_estimated_time(self, balance: dict[uuid.UUID, int]) -> float:
        time_table = self.convert_num_tasks_to_time(balance)
        time_list = [t for id_, t in time_table.items() if np.isnan(t)]
        return max(time_list) if len(time_list) > 0 else float("nan")

    def _get_adjacent_performance(self, id_: uuid.UUID, num_task: int) -> tuple[float, float, float]:
        if id_ not in self.performance_table:
            return float("nan"), float("nan"), float("nan")
        perf = self.performance_table[id_]
        return perf.get(num_task - 1), perf.get(num_task), perf.get(num_task + 1)

