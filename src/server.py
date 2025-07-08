"""
Optimization Server

This script starts an optimization server that distributes CMA-ES optimization
tasks to connected clients.
"""

import os
import threading
from datetime import datetime
from icecream import ic
from framework.optimization import OptimizationServer, CMAES

from controller import RobotNeuralNetwork
from settings import MySettings

# Configure icecream for distributed system debugging
ic.configureOutput(
    prefix=lambda: f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] SERVER| ',
    includeContext=True
)

ic.disable()

_generation = None


def handler(cmaes: CMAES):
    global _generation

    if _generation is not None and cmaes.generation == _generation:
        return

    _generation = cmaes.generation
    individuals = cmaes.individuals
    n = len(individuals)

    fittness = [individuals.get_fitness(i) for i in range(len(individuals))]
    ave_fittness = sum(fittness) / n

    print(f"Generation: {cmaes.generation} | Average: {ave_fittness:.2f}")


def main():
    dim = RobotNeuralNetwork().dim

    settings = MySettings()

    settings.Server.HOST = "0.0.0.0"
    settings.Optimization.dimension = dim

    print("=" * 50)
    print("OPTIMIZATION SERVER")
    print("=" * 50)
    print(f"Host: {settings.Server.HOST}")
    print(f"Port: {settings.Server.PORT}")
    print(f"Socket Backlog: {settings.Server.SOCKET_BACKLOG}")
    print("-" * 30)
    print(f"Problem dimension: {settings.Optimization.dimension}")
    print(f"Initial sigma: {settings.Optimization.sigma}")
    print(f"Population size: {settings.Optimization.population_size}")
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
