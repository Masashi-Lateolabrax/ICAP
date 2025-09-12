import argparse
import os
import threading
import time
from datetime import datetime
import math

from icecream import ic

from framework.prelude import Settings, Individual
from framework.optimization import connect_to_server

from config import Simulator

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
        
        fitness = backend.calc_total_score()
        max_gas_pheromone = backend.get_total_gas_pheromone()
        
        return fitness, max_gas_pheromone


def main(settings: Settings):
    parser = argparse.ArgumentParser(description="ICAP Optimization Client")
    parser.add_argument("--host", type=str, help="Server host address")
    parser.add_argument("--port", type=int, help="Server port number")
    parser.add_argument("--num-processes", type=int, default=1, help="Number of evaluation processes")
    args = parser.parse_args()

    host = args.host if args.host is not None else settings.Server.HOST
    port = args.port if args.port is not None else settings.Server.PORT

    print("=" * 50)
    print("OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {host}:{port}")
    print("-" * 30)
    print("Connecting to server...")
    print("Press Ctrl+C to disconnect")
    print("=" * 50)

    handler = Handler()
    evaluator = Evaluator(settings)

    connect_to_server(
        host,
        port,
        evaluation_function=evaluator.run,
        handler=handler.run,
        num_processes=args.num_processes,
    )


if __name__ == "__main__":
    main(
        Settings()
    )
