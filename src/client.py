"""
Optimization Client

This script connects to an optimization server to evaluate individuals
using CMA-ES optimization in a distributed manner.
"""
import argparse
import logging
import os
import threading
import time
import math
from datetime import datetime

import jax
from icecream import ic

from framework.prelude import Settings, TaskProgress
from framework.tasks import SharedTaskManager

from config import Simulator, Controller

# Configure icecream for distributed system debugging
ic.configureOutput(
    prefix=lambda: f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] CLIENT| ',
    includeContext=True
)

ic.disable()


class Handler:
    def __init__(self):
        self.time = -1

    def run(self, individuals: list[Individual]):
        current_time = time.time()
        throughput = len(individuals) / ((current_time - self.time) + 1e-10)
        self.time = current_time

        ave_fitness = sum([i.get_fitness() for i in individuals]) / len(individuals)

        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}] "
            f"num: {len(individuals)} "
            f"fitness:{ave_fitness} "
            f"throughput:{throughput:.2f} ind/s"
        )


class Evaluator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, individual: Individual):
        backend = Simulator(self.settings, individual, render=False)
        for _ in range(math.ceil(self.settings.Simulation.TIME_LENGTH / self.settings.Simulation.TIME_STEP)):
            backend.step()
        return backend.calc_total_score()


def main(settings: Settings):
    parser = argparse.ArgumentParser(description="ICAP Optimization Client")
    parser.add_argument("--host", type=str, help="Server host address")
    parser.add_argument("--port", type=int, help="Server port number")
    parser.add_argument("--num-processes", type=int, default=1, help="Number of evaluation processes")
    args = parser.parse_args()

    if not args.host:
        print("Error: --host argument is required")
        exit(1)

    if not args.port:
        print("Error: --port argument is required")
        exit(1)

    host = args.host
    port = args.port

    print("=" * 50)
    print("OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {host}:{port}")
    print("-" * 30)
    print(f"Number of processes: {args.num_processes}")
    print("-" * 30)
    print("Connecting to server...")
    print("Press Ctrl+C to disconnect")
    print("=" * 50)

    handler = Handler()
    evaluator = Evaluator(settings)

    try:
        client_evaluation(host, port, evaluator, handler, args.num_processes)
    except Exception as e:
        logging.error(f"Failed to connect to server: {e}")
        exit(1)


def client_evaluation(host: str, port: int, evaluator: Evaluator, handler: Handler, num_processes: int):
    """Main client evaluation loop using SharedTaskManager"""
    from framework.tasks import SharedTaskManager
    
    shared_task_manager = SharedTaskManager()
    
    while True:
        try:
            # Sync with server to get tasks
            if not shared_task_manager.sync(host, port):
                logging.warning(f"Failed to sync with server {host}:{port}")
                time.sleep(5.0)
                continue
            
            # Take available tasks
            tasks = shared_task_manager.take_task(n=num_processes)
            if not tasks:
                time.sleep(1.0)
                continue
            
            # Evaluate tasks
            evaluated_individuals = []
            for task in tasks:
                try:
                    individual = Individual.from_parameter(task.parameter)
                    fitness = evaluator.run(individual)
                    individual.set_fitness(fitness)
                    evaluated_individuals.append(individual)
                    
                    # Mark task as completed
                    completed_task = task.replace(
                        result=fitness,
                        progress=TaskProgress.COMPLETED
                    )
                    shared_task_manager.tasks[task.id.content_hash] = completed_task
                    
                except Exception as e:
                    logging.error(f"Error evaluating task: {e}")
                    continue
            
            # Report results via handler
            if evaluated_individuals and handler:
                handler.run(evaluated_individuals)
                
        except KeyboardInterrupt:
            logging.info("Client interrupted by user")
            break
        except Exception as e:
            logging.error(f"Client error: {e}")
            time.sleep(5.0)
            continue


def main(settings: Settings):
    parser = argparse.ArgumentParser(description="ICAP Optimization Client")
    parser.add_argument("--host", type=str, help="Server host address")
    parser.add_argument("--port", type=int, help="Server port number")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of tasks to evaluate in batch")
    args = parser.parse_args()

    if not args.host:
        print("Error: --host argument is required")
        exit(1)

    if not args.port:
        print("Error: --port argument is required")
        exit(1)

    host = args.host
    port = args.port

    print("=" * 50)
    print("OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {host}:{port}")
    print("-" * 30)
    print(f"Batch size: {args.batch_size}")
    print("-" * 30)
    print("Connecting to server...")
    print("Press Ctrl+C to disconnect")
    print("=" * 50)

    try:
        client_evaluation(host, port, settings, args.batch_size)
    except Exception as e:
        logging.error(f"Failed to connect to server: {e}")
        exit(1)


if __name__ == "__main__":
    from settings import MySettings

    main(
        MySettings()
    )