import uuid

from scipy import optimize
import numpy as np


class Performance:
    def __init__(self):
        self.performance_time: dict[int, float] = {}

    def has_enough_data(self) -> bool:
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
    log_processing_times = np.log1p(processing_times)
    return log_processing_times * np.ceil(num_tasks / num_allocated_tasks)


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

        res = {i: 1 for i, perfs in self.performance_table.items() if not perfs.has_enough_data()}
        remaining_tasks = total_tasks - len(res)

        if remaining_tasks <= 0:
            return res

        sufficient_data_pcs = {i: perf for i, perf in self.performance_table.items() if perf.has_enough_data()}

        if len(sufficient_data_pcs) == 0:
            return res

        optimal_allocation = self._optimize_remaining_tasks(remaining_tasks, sufficient_data_pcs)

        if optimal_allocation is None:
            pc_ids = list(sufficient_data_pcs.keys())
            for i in range(remaining_tasks):
                res[pc_ids[i % len(pc_ids)]] += 1
            return res

        for pc_id, count in optimal_allocation.items():
            res[pc_id] = res.get(pc_id, 0) + count

        return res

    def _optimize_remaining_tasks(
            self, remaining_tasks: int, sufficient_data_pcs: dict[uuid.UUID, Performance]
    ) -> dict[uuid.UUID, int] | None:

        pc_ids = list(sufficient_data_pcs.keys())
        n_pcs = len(pc_ids)

        if n_pcs == 0:
            return None
        if remaining_tasks <= 0:
            raise ValueError("remaining_tasks must be positive")
        for pc_id, perf in sufficient_data_pcs.items():
            if np.isnan(perf.get(1)):
                return None

        def objective(x):
            allocation_ = {}
            for i in range(n_pcs):
                task_count = max(0, int(round(x[i])))
                allocation_[pc_ids[i]] = task_count
            return _objective_func(remaining_tasks, sufficient_data_pcs, allocation_)

        def constraint(x):
            return sum(x) - remaining_tasks

        perf_weights = [1.0 / sufficient_data_pcs[pc_id].get(1) for pc_id in pc_ids]
        weight_sum = sum(perf_weights)
        x0 = [remaining_tasks * w / weight_sum for w in perf_weights]

        result = optimize.minimize(
            objective, x0,
            method='trust-constr',
            constraints={'type': 'eq', 'fun': constraint},
            bounds=[(0, remaining_tasks) for _ in range(n_pcs)]
        )

        if result.success and result.fun != float('inf'):
            allocation = {pc_ids[i]: max(0, result.x[i]) for i in range(n_pcs)}

            while True:
                current_total = sum([int(v) for v in allocation.values()])
                diff = remaining_tasks - current_total
                if diff == 0:
                    break

                fractional_parts = {k: v - int(v) for k, v in allocation.items() if v - int(v) > 0}
                if len(fractional_parts) == 0:
                    break

                if diff > 0:
                    max_pc = max(fractional_parts, key=lambda pc: fractional_parts[pc])
                    allocation[max_pc] = int(allocation[max_pc] + 1)
                    continue
                else:
                    min_pc = min(fractional_parts, key=lambda pc: fractional_parts[pc])
                    if allocation[min_pc] > 0:
                        allocation[min_pc] = max(int(allocation[min_pc] - 1), 0)
                    continue

            return {pc_id: int(v) for pc_id, v in allocation.items()}
        else:
            return None

    def record_task_performance(self, worker_id: uuid.UUID, task_count: int, processing_time: float):
        if worker_id in self.performance_table:
            self.performance_table[worker_id].register(task_count, processing_time)
