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