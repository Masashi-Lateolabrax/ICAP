import os
import threading
from datetime import datetime

from icecream import ic
from framework.optimization import run_adaptive_client_manager

from evaluation_function import evaluation_function
from framework.prelude import Individual

ic.configureOutput(
    prefix=lambda: f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] PARALLEL| ',
    includeContext=True
)

ic.enable()


def handler(individual: Individual):
    throughput = 1 / (individual.get_elapse() + 1e-10)
    print(
        f"pid:{os.getpid()} "
        f"fitness:{individual.get_fitness()} "
        f"throughput:{throughput:.2f} ind/s"
    )


def main():
    from settings import MySettings
    settings = MySettings()

    print("=" * 50)
    print("PARALLEL OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {settings.Server.HOST}:{settings.Server.PORT}")
    print("-" * 30)
    print("Starting adaptive client manager...")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    run_adaptive_client_manager(
        host=settings.Server.HOST,
        port=settings.Server.PORT,
        evaluation_function=evaluation_function,
        adjustment_interval=5,
        observation_interval=10
    )


if __name__ == "__main__":
    main()
