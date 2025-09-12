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
import asyncio

import numpy as np
from icecream import ic
from cmaes import CMA

from framework.prelude import *
from framework.tasks.network_manager import NetworkServer

from config import Controller

# Configure icecream for distributed system debugging
ic.configureOutput(
    prefix=lambda: f'[{datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] SERVER| ',
    includeContext=True
)


# ic.disable()  # Enable debugging


def get_git_hash() -> str:
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, cwd=os.path.dirname(__file__))
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception as e:
        logging.error(f"Failed to get git hash: {e}")
        return "unknown"


def create_optimization_result(settings: Settings, generation: int, completed_tasks) -> OptimizationResult:
    ic(generation, len(completed_tasks))
    num_to_save = max(1, settings.Storage.TOP_N) if settings.Storage.TOP_N > 0 else len(completed_tasks)
    tasks_to_save = sorted(completed_tasks, key=lambda x: x[1])[:num_to_save]
    ic(num_to_save, len(tasks_to_save), settings.Storage.TOP_N)
    result = OptimizationResult.new(generation, tasks_to_save)
    ic(result.generation, result.avg_fitness)
    return result


def save_completed_tasks(settings: Settings, result: OptimizationResult) -> OptimizationResult:
    ic(settings.Storage.SAVE_INDIVIDUALS)
    if not settings.Storage.SAVE_INDIVIDUALS:
        return result

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    git_hash = get_git_hash()
    filename = f"generation_{result.generation}.pkl"
    folder_name = f"{timestamp}_{git_hash}"

    save_directory = os.path.join(settings.Storage.SAVE_DIRECTORY, folder_name)
    file_path = os.path.join(save_directory, filename)

    ic(file_path)
    os.makedirs(save_directory, exist_ok=True)

    try:
        ic(result.save(file_path))
    except Exception as e:
        ic(e)
        raise RuntimeError(f"Failed to save completed tasks: {e}")

    return result


def print_info(
        settings: Settings,
        prev_result: Optional[OptimizationResult],
        current_result: OptimizationResult
):
    current_time = datetime.datetime.now()

    if prev_result is not None:
        time_diff = current_result.timestamp - prev_result.timestamp
        speed = time_diff.total_seconds()  # sec/gen

        remaining_generations = settings.Optimization.GENERATION - current_result.generation
        remaining_seconds = datetime.timedelta(seconds=remaining_generations * speed)
        eta = current_time + remaining_seconds

    else:
        speed = float("nan")
        eta = "N/A"

    print(
        f"[{current_time.strftime('%H:%M:%S')}] "
        f"Generation: {current_result.generation}, | "
        f"Average: {current_result.avg_fitness:.2f} | "
        f"SD: {math.sqrt(current_result.variance):.2f} | "
        f"Speed: {settings.Optimization.POPULATION / speed:.2f} ind/sec | "
        f"ETA: {eta} "
    )


def print_and_save(settings: Settings, prev_result: Optional[OptimizationResult], current_result: OptimizationResult):
    print_info(settings, prev_result, current_result)
    save_completed_tasks(settings, current_result)


async def optimization(port: int, timeout: int, settings: Settings):
    ic(port, timeout, settings.Optimization.GENERATION)
    optimization_result = None

    async with NetworkServer('localhost', port, timeout) as server:
        shared_task_manager = server.task_manager
        ic(shared_task_manager)

        dim = Controller.dim()
        ic(dim)
        cmaes = CMA(
            mean=np.zeros(dim, dtype=np.float32),
            sigma=settings.Optimization.SIGMA,
            seed=settings.Optimization.SEED,
            population_size=settings.Optimization.POPULATION,
        )
        ic(settings.Optimization.SIGMA, settings.Optimization.POPULATION, settings.Optimization.SEED)

        for i in range(settings.Optimization.GENERATION):
            ic(i, settings.Optimization.GENERATION)

            # Generate tasks and distribute them to clients
            parameters = [cmaes.ask() for _ in range(settings.Optimization.POPULATION)]
            ic(len(parameters), parameters[0].shape if parameters else "no_params")

            tasks = [Task.new(None, x, i) for x in parameters]
            for task in tasks:
                shared_task_manager.set_task(task)
            ic(len(tasks), len(shared_task_manager))

            completed_tasks = []
            ic(len(shared_task_manager))

            while len(shared_task_manager) > 0:
                # Wait a short time for tasks to be processed by clients
                await asyncio.sleep(1.0)

                # Retrieve completed tasks
                newly_completed = shared_task_manager.retrieve_completed_tasks()
                completed_tasks += newly_completed

                if newly_completed:
                    ic(len(newly_completed), len(completed_tasks), len(shared_task_manager))

                    # Show task status for debugging
                    status = shared_task_manager.get_task_status()
                    ic(status)

            ic(len(completed_tasks), i)

            # Create optimization result
            prev_optimization_result = optimization_result
            optimization_result = create_optimization_result(settings, i, completed_tasks)
            ic(optimization_result.avg_fitness, optimization_result.variance)

            # Update CMA-ES with completed tasks
            fitness: list[tuple[np.ndarray, float]] = [(task.parameter, task.result) for task in completed_tasks]
            ic(cmaes.tell(fitness))

            print_and_save(settings, prev_optimization_result, optimization_result)


def optimization(port: int, timeout: int, settings: Settings):
    shared_task_manager = SharedTaskManager()
    shared_task_manager.start_listening(port, timeout)

    dim = Controller.dim()
    cmaes = CMA(
        mean=np.zeros(dim, dtype=np.float32),
        sigma=settings.Optimization.SIGMA,
        population_size=settings.Optimization.POPULATION,
        max_generation=settings.Optimization.MAX_GENERATION
    )


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

    asyncio.run(optimization(port, 60, settings))


if __name__ == "__main__":
    main(
        Settings()
    )
