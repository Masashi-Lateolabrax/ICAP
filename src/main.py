import math
import os
import threading
from datetime import datetime

from icecream import ic
from framework.prelude import *
from framework.optimization import connect_to_server

from settings import MySettings
from simulator import Simulator

# Configure icecream for distributed system debugging
ic.configureOutput(
    prefix=lambda: f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] CLIENT| ',
    includeContext=True
)

ic.disable()


def evaluation_function(individual: Individual):
    settings = MySettings()

    RobotValues.set_max_speed(settings.Robot.MAX_SPEED)
    RobotValues.set_distance_between_wheels(settings.Robot.DISTANCE_BETWEEN_WHEELS)
    RobotValues.set_robot_height(settings.Robot.HEIGHT)

    backend = Simulator(settings, individual)
    for _ in range(math.ceil(settings.Simulation.TIME_LENGTH / settings.Simulation.TIME_STEP)):
        backend.step()

    return backend.total_score()


def handler(individual: Individual):
    throughput = 1 / (individual.get_elapse() + 1e-10)
    print(
        f"pid:{os.getpid()} "
        f"fitness:{individual.get_fitness()} "
        f"throughput:{throughput:.2f} ind/s"
    )


def main():
    settings = MySettings()

    print("=" * 50)
    print("OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {settings.Server.HOST}:{settings.Server.PORT}")
    print("-" * 30)
    print("Connecting to server...")
    print("Press Ctrl+C to disconnect")
    print("=" * 50)

    connect_to_server(
        settings.Server.HOST,
        settings.Server.PORT,
        evaluation_function=evaluation_function,
        handler=handler,
    )


if __name__ == "__main__":
    main()
