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

    def retrieve_finished_tasks(self) -> list[Task]:
        return [task for task in self.tasks if task.is_finished()]
