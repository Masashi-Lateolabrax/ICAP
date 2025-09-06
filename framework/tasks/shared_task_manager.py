import datetime
import random

from ..prelude import *


class SharedTaskManager:
    def __init__(self):
        self.tasks: list[Task] = []

    def listen_(self, port: int):
        pass

    def sync_(self, target_addr, port):
        pass

    def take_task_(self, n: int = 1, deadline: int = 300) -> list[Task]:
        current_time = datetime.datetime.now(datetime.UTC)

        # Reset tasks that have been running too long
        running_task_keys = [key for key, task in self.tasks.items() if task.progress.is_running()]
        for key in running_task_keys:
            task = self.tasks[key]
            if (current_time - task.timestamp).total_seconds() > deadline:
                self.tasks[key] = task.replace(progress=TaskProgress.WAITING)

        # Select waiting tasks randomly
        waiting_task_keys = [key for key, task in self.tasks.items() if task.progress.is_waiting()]
        random.shuffle(waiting_task_keys)
        num = len(waiting_task_keys)
        waiting_task_keys = waiting_task_keys[:min(n, num)]

        # Mark them as running
        waiting_tasks = []
        for key in waiting_task_keys:
            replaced_task = self.tasks[key].replace(progress=TaskProgress.RUNNING)
            self.tasks[key] = replaced_task
            waiting_tasks.append(replaced_task)

        return waiting_tasks

    def add_task_(self, task: Task):
        self.tasks[task.id.content_hash] = task

    def retrieve_completed_tasks(self) -> list[Task]:
        result = []
        n = len(self.tasks)
        for i in range(n - 1, -1, -1):
            if self.tasks[i].is_completed():
                result.append(self.tasks.pop(i))
        return result
