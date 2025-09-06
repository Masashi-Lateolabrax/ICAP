import datetime
import logging
import random

from ..prelude import *
from .network_manager import NetworkManager


class SharedTaskManager:
    def __init__(self):
        self._tasks: dict[bytes, Task] = {}  # key: TaskID.content_hash, value: Task
        self._network_manager = NetworkManager()

    def _update(self, target_tasks: dict[bytes, Task], self_is_priority: bool):
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

    def start_communication(self, port: int, timeout: int = 30) -> bool:
        """Start listening on port for incoming connections."""
        return self._network_manager.start_communication(port, timeout)

    def stop_communication(self):
        """Stop listening and close server socket."""
        self._network_manager.stop_communication()

    def listen(self):
        """Listen for incoming UDP packets and exchange tasks."""
        result = self._network_manager.receive_tasks()
        if result is None:
            return

        incoming_tasks, address = result
        logging.info(f"Received {len(incoming_tasks)} tasks from {address[0]}:{address[1]}")
        self._update(incoming_tasks, True)

        # Send our tasks back to the client
        self._network_manager.send_tasks(self._tasks, address)

    def sync(self, target_addr: str, port: int) -> bool:
        """Send tasks to target."""
        try:
            self._network_manager.send_tasks(self._tasks, (target_addr, port))
            return True
        except Exception as e:
            logging.error(f"Failed to send tasks to {target_addr}:{port}: {e}")
            return False

    def take_task(self, n: int = 1, deadline: int = 300) -> list[Task]:
        current_time = datetime.datetime.now(datetime.UTC)

        # Reset tasks that have been running too long
        running_task_keys = [key for key, task in self._tasks.items() if task.progress.is_running()]
        for key in running_task_keys:
            task = self._tasks[key]
            running_time = (current_time - task.timestamp).total_seconds()
            if running_time > deadline:
                self._tasks[key] = task.replace(progress=TaskProgress.WAITING)
                logging.warning(f"Task {task.id.content_hash[:8]} timed out after {running_time:.1f}s")

        # Select waiting tasks randomly
        waiting_task_keys = [key for key, task in self._tasks.items() if task.progress.is_waiting()]
        random.shuffle(waiting_task_keys)
        num = len(waiting_task_keys)
        waiting_task_keys = waiting_task_keys[:min(n, num)]

        # Mark them as running
        waiting_tasks = []
        for key in waiting_task_keys:
            replaced_task = self._tasks[key].replace(progress=TaskProgress.RUNNING)
            self._tasks[key] = replaced_task
            waiting_tasks.append(replaced_task)

        return waiting_tasks

    def set_task(self, task: Task):
        self._tasks[task.id.content_hash] = task

    def retrieve_completed_tasks(self) -> list[Task]:
        completed_tasks = [task for task in self._tasks.values() if task.progress.is_completed()]
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

    def __del__(self):
        """Cleanup socket on destruction."""
        self.stop_communication()

    def __len__(self):
        not_completed_tasks = [task for task in self._tasks.values() if not task.progress.is_completed()]
        return len(not_completed_tasks)
