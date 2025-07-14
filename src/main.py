import argparse
import os
import threading
import time
from datetime import datetime

from icecream import ic
import numpy as np

from framework.prelude import Settings, Individual
from framework.interfaces import SensorInterface
from framework.sensor import PreprocessedOmniSensor, DirectionSensor
from framework.optimization import connect_to_server

from src import utils
from settings import MySettings

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
            f"pid:{os.getpid()} "
            f"num: {len(individuals)} "
            f"fitness:{ave_fitness} "
            f"throughput:{throughput:.2f} ind/s"
        )


def main():
    parser = argparse.ArgumentParser(description="ICAP Optimization Client")
    parser.add_argument("--host", type=str, help="Server host address")
    parser.add_argument("--port", type=int, help="Server port number")
    args = parser.parse_args()

    settings = MySettings()

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

    connect_to_server(
        host,
        port,
        evaluation_function=EvaluationFunction(
            settings, lambda ind: Simulator(settings, ind)
        ).run,
        handler=handler.run,
    )


if __name__ == "__main__":
    main()
