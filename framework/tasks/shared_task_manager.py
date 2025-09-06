import datetime
import logging
import random
from typing import Optional

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
        if self.is_listening:
            logging.warning("Already listening")
            return False

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.settimeout(timeout)
            self.server_socket.bind(('0.0.0.0', port))
            self.server_socket.listen(5)
            self.is_listening = True
            logging.info(f"TaskManager started listening on port {port} (timeout: {timeout}s)")
            return True
        except Exception as e:
            logging.error(f"Failed to start listening: {e}")
            return False

    def stop_listening(self):
        """Stop listening and close server socket."""
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        self.is_listening = False
        logging.info("Stopped listening")

    def listen_(self, port: int, timeout: int):
        pass  # TODO: listen on port for other task managers
        tasks_in_sender: dict[bytes, Task] = None  # TODO: get tasks from other task manager
        self._update(tasks_in_sender, True)

    def sync_(self, target_addr, port) -> bool:  # returns False if connection failed
        pass  # TODO: connect to other task manager
        tasks_in_target: dict[bytes, Task] = None  # TODO: get tasks from other task manager
        self._update(tasks_in_target, False)
        return True

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
