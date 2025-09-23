class Performance:
    def __init__(self):
        self.performance: dict[int, float] = {}

    def register(self, task_count: int, processing_time: float, alpha: float = 0.2):
        current = self.performance.get(task_count, 0.0)
        self.performance[task_count] = (1 - alpha) * current + alpha * processing_time

    def get(self, task_count: int) -> float:
        if task_count == 0:
            return 0.0
        return self.performance.get(task_count, float("nan"))


class LoadBalancer:
    def __init__(self):
        self.performance_table: dict[uuid.UUID, Performance] = {}
        self._buf_table: dict[uuid.UUID, int] = {}

    def remove(self, id_: uuid.UUID):
        if id_ in self.performance_table:
            del self.performance_table[id_]

    def register_performance(self, id_: uuid.UUID, task_count: int, performance: float):
        if id_ not in self.performance_table:
            self.performance_table[id_] = Performance()
        self.performance_table[id_].register(task_count, performance)
