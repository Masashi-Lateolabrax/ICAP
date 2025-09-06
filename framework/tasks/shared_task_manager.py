import datetime
import random

from ..prelude import *


class SharedTaskManager:
    def __init__(self):
        self.tasks: dict[bytes, Task] = {}  # key: TaskID.content_hash, value: Task


    def listen_(self, port: int):
        pass

    def sync_(self, target_addr, port):
        pass

    def take_task(self, n: int = 1, deadline: int = 300) -> list[Task]:
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

    def add_task(self, task: Task):
        if task.id.content_hash in self.tasks:
            logging.warning(f"Task {task.id.content_hash.hex()} already exists. Skipping addition.")
            return
        self.tasks[task.id.content_hash] = task

    def retrieve_completed_tasks(self) -> list[Task]:
        completed_tasks = [task for task in self.tasks.values() if task.progress.is_completed()]
        for task in completed_tasks:
            self.tasks.pop(task.id.content_hash, None)
        return completed_tasks
