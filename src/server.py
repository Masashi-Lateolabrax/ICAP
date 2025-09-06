"""
Optimization Server

This script starts an optimization server that distributes CMA-ES optimization
tasks to connected clients.
"""
import argparse
import logging
import os
import threading
import datetime
import subprocess
from typing import Optional
import math

import numpy as np
from icecream import ic
from cmaes import CMA

from framework.prelude import *
from framework.tasks import SharedTaskManager

from config import Controller

# Configure icecream for distributed system debugging
ic.configureOutput(
    prefix=lambda: f'[{datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] SERVER| ',
    includeContext=True
)

ic.disable()


def get_git_hash() -> str:
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, cwd=os.path.dirname(__file__))
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_completed_tasks(settings: Settings, generation: int, completed_tasks: list[Task]) -> OptimizationResult:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    git_hash = get_git_hash()
    filename = f"generation_{generation}.pkl"
    folder_name = f"{timestamp}_{git_hash}"

    save_directory = os.path.join(settings.Storage.SAVE_DIRECTORY, folder_name)
    file_path = os.path.join(save_directory, filename)

    os.makedirs(save_directory, exist_ok=True)

    num_to_save = max(1, settings.Storage.TOP_N) if settings.Storage.TOP_N > 0 else len(completed_tasks)
    tasks_to_save = sorted(completed_tasks, key=lambda x: x[1])[:num_to_save]

    try:
        result = OptimizationResult.new(generation, tasks_to_save)
        result.save(file_path)

    except Exception as e:
        print(f"Error saving individuals: {e}")


def optimization(port: int, timeout: int, settings: Settings):
    shared_task_manager = SharedTaskManager()
    shared_task_manager.start_listening(port, timeout)

    dim = Controller.dim()
    cmaes = CMA(
        mean=np.zeros(dim, dtype=np.float32),
        sigma=settings.Optimization.SIGMA,
        seed=settings.Optimization.SEED,
        population_size=settings.Optimization.POPULATION,
    )

    for i in range(settings.Optimization.GENERATION):
        # Generate tasks and distribute them to clients
        for task in [Task.new(None, x, i) for x in cmaes.ask()]:
            shared_task_manager.add_task(task)

        completed_tasks = []
        while len(shared_task_manager) > 0:
            # Synchronize task manager
            shared_task_manager.listen()

            # Retrieve completed tasks
            completed_tasks += shared_task_manager.retrieve_completed_tasks()

        # Update CMA-ES with completed tasks
        fitness: list[tuple[np.ndarray, float]] = [(task.parameter, task.result) for task in completed_tasks]
        cmaes.tell(fitness)

        print_and_save(settings, i, cmaes, completed_tasks)


def main(settings: Settings):
    parser = argparse.ArgumentParser(description="ICAP Optimization Server")
    parser.add_argument("--port", type=int, help="Server port number")
    args = parser.parse_args()

    if not args.port:
        print("Error: --port argument is required")
        exit(1)

    port = args.port

    print("=" * 50)
    print("OPTIMIZATION SERVER")
    print("=" * 50)
    print(f"Port: {port}")
    print("-" * 30)
    print(f"Problem dimension: {Controller.dim()}")
    print(f"Initial sigma: {settings.Optimization.SIGMA}")
    print(f"Population size: {settings.Optimization.POPULATION}")
    print("-" * 30)
    print(f"Save individuals: {settings.Storage.SAVE_INDIVIDUALS}")
    if settings.Storage.SAVE_INDIVIDUALS:
        print(f"Save directory: {settings.Storage.SAVE_DIRECTORY}")
        print(f"Save interval: {settings.Storage.SAVE_INTERVAL} generations")
    print("-" * 30)
    print("Waiting for clients to connect...")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

    optimization(port, settings)


if __name__ == "__main__":
    from settings import MySettings

    main(
        MySettings()
    )
