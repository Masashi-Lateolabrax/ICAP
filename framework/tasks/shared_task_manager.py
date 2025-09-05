from ..prelude import *


class SharedTaskManager:
    def __init__(self):
        self.tasks: list[Task] = []

    def listen_(self, port: int):
        pass

    def sync_(self, target_addr, port):
        pass

    def take_task_(self):
        pass

    def add_task_(self, task: Task):
        self.tasks.append(task)

    def retrieve_completed_tasks(self) -> list[Task]:
        result = []
        n = len(self.tasks)
        for i in range(n - 1, -1, -1):
            if self.tasks[i].is_completed():
                result.append(self.tasks.pop(i))
        return result
