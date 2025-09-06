import datetime
import logging
import random

from ..prelude import *
from .network_manager import NetworkManager


class SharedTaskManager:
    def __init__(self):
        self.tasks: dict[bytes, Task] = {}  # key: TaskID.content_hash, value: Task
        self.network_manager = NetworkManager()

    def _update(self, target_tasks: dict[bytes, Task], self_is_priority: bool):
        for target_key, target_task in target_tasks.items():
            if target_key not in self.tasks:
                if not self_is_priority:
                    self.tasks[target_key] = target_task
                continue

            my_task = self.tasks[target_key]
            if my_task.progress.is_completed():
                continue
            if my_task.fingerprint == target_task.fingerprint:
                continue

            if target_task.timestamp > my_task.timestamp:
                self.tasks[target_key] = target_task

    def start_listening(self, port: int, timeout: int = 30) -> bool:
        """Start listening on port for incoming connections."""
        return self.network_manager.start_listening(port, timeout)

    def stop_listening(self):
        """Stop listening and close server socket."""
        self.network_manager.stop_listening()

    def listen(self):
        """Listen for incoming UDP packets and exchange tasks."""
        result = self.network_manager.receive_tasks()
        if result is None:
            return

        incoming_tasks, address = result
        logging.info(f"Received {len(incoming_tasks)} tasks from {address[0]}:{address[1]}")
        self._update(incoming_tasks, True)

        # Send our tasks back to the client
        self.network_manager.send_tasks(self.tasks, address)

    def sync(self, target_addr: str, port: int) -> bool:
        """Send tasks to target."""
        try:
            self.network_manager.send_tasks(self.tasks, (target_addr, port))
            return True
        except Exception as e:
            logging.error(f"Failed to send tasks to {target_addr}:{port}: {e}")
            return False

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

    def __del__(self):
        """Cleanup socket on destruction."""
        self.stop_listening()

    def __len__(self):
        not_completed_tasks = [task for task in self.tasks.values() if not task.progress.is_completed()]
        return len(not_completed_tasks)
