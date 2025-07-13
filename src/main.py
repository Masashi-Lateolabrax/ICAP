import os
import threading
import time
from datetime import datetime

from icecream import ic
from framework.optimization import connect_to_server

from evaluation_function import evaluation_function
from framework.prelude import Individual

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
        throughput = 1 / ((current_time - self.time) + 1e-10)
        self.time = current_time

        ave_fitness = sum([i.get_fitness() for i in individuals]) / len(individuals)

        print(
            f"pid:{os.getpid()} "
            f"num: {len(individuals)} "
            f"fitness:{ave_fitness} "
            f"throughput:{throughput:.2f} ind/s"
        )


def main():
    from settings import MySettings
    settings = MySettings()

    print("=" * 50)
    print("OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {settings.Server.HOST}:{settings.Server.PORT}")
    print("-" * 30)
    print("Connecting to server...")
    print("Press Ctrl+C to disconnect")
    print("=" * 50)

    handler = Handler()

    connect_to_server(
        settings.Server.HOST,
        settings.Server.PORT,
        evaluation_function=evaluation_function,
        handler=handler.run,
    )


if __name__ == "__main__":
    main()
