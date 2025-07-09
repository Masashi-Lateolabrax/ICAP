"""
Optimization Server

This script starts an optimization server that distributes CMA-ES optimization
tasks to connected clients.
"""

import os
import threading
import datetime
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
    print("Waiting for clients to connect...")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

    server = OptimizationServer(
        settings,
        handler=handler
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
