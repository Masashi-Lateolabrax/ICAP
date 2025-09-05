from typing import Optional

from ..prelude import *


class SharedTaskManager:
    def __init__(self):
        self.tasks: list[Task] = []

    def listen_(self, port: int):
        pass

    def sync_(self, target_addr, port):
        pass

    def take_task_(self, n: int = 1) -> list[Task]:
        result: list[Task] = []
        for task in self.tasks:
            if task.is_waiting():
                result.append(task)
                if len(result) >= n:
                    return result
        return []

    def add_task_(self, task: Task):
        self.tasks.append(task)

    def retrieve_completed_tasks(self) -> list[Task]:
        result = []
        n = len(self.tasks)
        for i in range(n - 1, -1, -1):
            if self.tasks[i].is_completed():
                result.append(self.tasks.pop(i))
        return result
