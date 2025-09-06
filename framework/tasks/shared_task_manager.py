import datetime
import logging
import random

from icecream import ic
from ..prelude import *


class SharedTaskManager:
    def __init__(self):
        self._tasks: dict[bytes, Task] = {}  # key: TaskID.content_hash, value: Task

    def update(self, target_tasks: dict[bytes, Task], self_is_priority: bool):
        ic(len(target_tasks), self_is_priority, len(self._tasks))
        for target_key, target_task in target_tasks.items():
            if target_key not in self._tasks:
                if not self_is_priority:
                    self._tasks[target_key] = target_task
                continue

            my_task = self._tasks[target_key]
            if my_task.progress.is_completed():
                continue
            if my_task.fingerprint == target_task.fingerprint:
                continue

            if target_task.timestamp > my_task.timestamp:
                self._tasks[target_key] = target_task

    def take_task(self, n: int = 1, deadline: int = 300) -> list[Task]:
        ic(n, deadline)
        current_time = datetime.datetime.now(datetime.UTC)

        # Reset tasks that have been running too long
        running_task_keys = [key for key, task in self._tasks.items() if task.progress.is_running()]
        ic(len(running_task_keys))
        for key in running_task_keys:
            task = self._tasks[key]
            running_time = (current_time - task.timestamp).total_seconds()
            if running_time > deadline:
                ic(task.id.content_hash[:8], running_time)
                self._tasks[key] = task.replace(progress=TaskProgress.WAITING)
                logging.warning(f"Task {task.id.content_hash[:8]} timed out after {running_time:.1f}s")

        # Select waiting tasks randomly
        waiting_task_keys = [key for key, task in self._tasks.items() if task.progress.is_waiting()]
        random.shuffle(waiting_task_keys)
        num = len(waiting_task_keys)
        ic(num, n)
        waiting_task_keys = waiting_task_keys[:min(n, num)]

        # Mark them as running
        waiting_tasks = []
        for key in waiting_task_keys:
            replaced_task = self._tasks[key].replace(progress=TaskProgress.RUNNING)
            self._tasks[key] = replaced_task
            waiting_tasks.append(replaced_task)

        ic(len(waiting_tasks))
        return waiting_tasks

    def set_task(self, task: Task):
        ic(task.id.content_hash[:8], task.progress, task.result if hasattr(task, 'result') else None)
        self._tasks[task.id.content_hash] = task

    def retrieve_completed_tasks(self) -> list[Task]:
        completed_tasks = [task for task in self._tasks.values() if task.progress.is_completed()]
        ic(len(completed_tasks))
        if completed_tasks:
            logging.info(f"Retrieving {len(completed_tasks)} completed tasks")
        for task in completed_tasks:
            self._tasks.pop(task.id.content_hash, None)
        return completed_tasks

    def get_task_status(self) -> dict[str, int]:
        """Get current task status counts for monitoring"""
        status_counts = {"waiting": 0, "running": 0, "completed": 0}
        for task in self._tasks.values():
            if task.progress.is_waiting():
                status_counts["waiting"] += 1
            elif task.progress.is_running():
                status_counts["running"] += 1
            elif task.progress.is_completed():
                status_counts["completed"] += 1
        return status_counts

    def __len__(self):
        not_completed_tasks = [task for task in self._tasks.values() if not task.progress.is_completed()]
        return len(not_completed_tasks)
