"""
Optimization Server

This script starts an optimization server that distributes CMA-ES optimization
tasks to connected clients.
"""

import os
import threading
import datetime
import pickle
from typing import Optional

from icecream import ic
from framework.prelude import *
from framework.optimization import OptimizationServer, CMAES

from controller import RobotNeuralNetwork
from settings import MySettings

# Configure icecream for distributed system debugging
ic.configureOutput(
    prefix=lambda: f'[{datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] SERVER| ',
    includeContext=True
)

ic.disable()

_generation = None
_last_call_time: Optional[datetime.datetime] = None


def save_individuals(cmaes: CMAES, individuals: list[Individual], settings: MySettings):
    try:
        os.makedirs(settings.Storage.SAVE_DIRECTORY, exist_ok=True)
        file_path = os.path.join(settings.Storage.SAVE_DIRECTORY, f"generation_{cmaes.generation:04d}.pkl")

        num_to_save = max(1, settings.Storage.TOP_N) if settings.Storage.TOP_N > 0 else len(individuals)
        sorted_individuals = sorted(individuals, key=lambda x: x.get_fitness())
        individuals_to_save = sorted_individuals[:num_to_save]

        with open(file_path, 'wb') as f:
            pickle.dump({
                'generation': cmaes.generation,
                'best_fitness': min(ind.get_fitness() for ind in individuals_to_save),
                'worst_fitness': max(ind.get_fitness() for ind in individuals),
                'avg_fitness': sum(ind.get_fitness() for ind in individuals) / len(individuals),
                'timestamp': datetime.datetime.now().isoformat(),
                "num_individuals": len(individuals_to_save),
                "individuals": individuals_to_save
            }, f)

        print(f"Saved {len(individuals_to_save)} individuals to {file_path}")

    except Exception as e:
        print(f"Error saving individuals: {e}")


def create_handler(settings: MySettings):
    def handler(cmaes: CMAES, individuals: list[Individual]):
        global _generation, _last_call_time

        current_time = datetime.datetime.now()

        ic(current_time, cmaes.generation, len(individuals))

        if _last_call_time is None:
            _last_call_time = current_time
            return
        if _generation is not None and cmaes.generation == _generation:
            return

        _generation = cmaes.generation
        time_diff = current_time - _last_call_time
        _last_call_time = current_time

        fittness = [i.get_fitness() for i in individuals]
        ave_fittness = sum(fittness) / len(fittness)

        speed = time_diff.total_seconds()  # sec/gen
        remaining_generations = cmaes.max_generation - cmaes.generation
        remaining_seconds = datetime.timedelta(seconds=remaining_generations * speed)

        print(
            f"[{current_time.strftime("%H:%M:%S")}] "
            f"Generation: {cmaes.generation} | "
            f"Average: {ave_fittness:.2f} | "
            f"Speed: {len(fittness) / speed:.2f} ind/sec | "
            f"ETA: {(current_time + remaining_seconds)} "
        )

        if settings.Storage.SAVE_INDIVIDUALS and cmaes.generation % settings.Storage.SAVE_INTERVAL == 0:
            save_individuals(cmaes, individuals, settings)

    return handler


def main():
    dim = RobotNeuralNetwork().dim

    settings = MySettings()

    settings.Server.HOST = "0.0.0.0"
    settings.Optimization.DIMENSION = dim

    print("=" * 50)
    print("OPTIMIZATION SERVER")
    print("=" * 50)
    print(f"Host: {settings.Server.HOST}")
    print(f"Port: {settings.Server.PORT}")
    print(f"Socket Backlog: {settings.Server.SOCKET_BACKLOG}")
    print("-" * 30)
    print(f"Problem dimension: {settings.Optimization.DIMENSION}")
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

    server = OptimizationServer(
        settings,
        handler=create_handler(settings)
    )

    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Server stopped by user")
        print("=" * 50)
    except Exception as e:
        print(f"\nServer error: {e}")


if __name__ == "__main__":
    main()
