import uuid
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


class Performance:
    def __init__(self):
        self.performance_time: dict[int, float] = {}

    def has_enugh_data(self) -> bool:
        return len(self.performance_time) >= 2

    def register(self, task_count: int, processing_time: float, alpha: float = 0.2):
        current = self.performance_time.get(task_count, 0.0)
        self.performance_time[task_count] = (1 - alpha) * current + alpha * processing_time

    def get(self, task_count: int) -> float:
        if task_count in self.performance_time:
            return self.performance_time[task_count]

        if len(self.performance_time) < 2:
            return float("nan")

        keys = np.array(list(self.performance_time.keys()))

        dists = np.abs(keys - task_count)
        indexes = np.argsort(dists)[:2]

        x1, x2 = int(keys[indexes[0]]), int(keys[indexes[1]])
        y1, y2 = self.performance_time[x1], self.performance_time[x2]

        if x1 == x2:
            return y1

        slope = (y2 - y1) / (x2 - x1)
        result = y1 + slope * (task_count - x1)

        return max(0.0, result)


def _objective_func(num_tasks: int, perf: dict[uuid.UUID, Performance], load_balance: dict[uuid.UUID, int]) -> float:
    if set(perf.keys()) != set(load_balance.keys()):
        raise ValueError("Keys of perf and load_balance must match")

    num_allocated_tasks = sum([n for n in load_balance.values()])
    if num_allocated_tasks == 0:
        raise ValueError("No tasks allocated in load_balance")

    processing_times = max([perf[i].get(n) for i, n in load_balance.items()])
    return processing_times * np.ceil(num_tasks / num_allocated_tasks)


class LoadBalancer:
    def __init__(self):
        self.performance_table: dict[uuid.UUID, Performance] = {}

    def remove(self, id_: uuid.UUID):
        if id_ in self.performance_table:
            del self.performance_table[id_]

    def register_performance(self, id_: uuid.UUID, task_count: int = 0, time: float = 0.0):
        if id_ not in self.performance_table:
            self.performance_table[id_] = Performance()
        self.performance_table[id_].register(task_count, time)

    def calc_balance(self, total_tasks: int) -> dict[uuid.UUID, int]:
        if len(self.performance_table) == 0:
            return {}

        res = {i: 1 for i, perfs in self.performance_table.items() if not perfs.has_enugh_data()}
        remaining_tasks = total_tasks - len(res)

        if remaining_tasks <= 0:
            return res

        sufficient_data_pcs = {i: perf for i, perf in self.performance_table.items() if perf.has_enugh_data()}

        if len(sufficient_data_pcs) == 0:
            return res

        optimal_allocation = self._optimize_remaining_tasks(remaining_tasks, sufficient_data_pcs)

        for pc_id, count in optimal_allocation.items():
            res[pc_id] = res.get(pc_id, 0) + count

        return res

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


    def record_task_performance(self, worker_id: uuid.UUID, task_count: int, processing_time: float):
        if worker_id in self.performance_table:
            self.performance_table[worker_id].register(task_count, processing_time)
