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
    prefix=lambda: f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] CLIENT| '
)


def evaluation_function(individual: Individual):
    settings = MySettings()

    RobotValues.set_max_speed(settings.Robot.MAX_SPEED)
    RobotValues.set_distance_between_wheels(settings.Robot.DISTANCE_BETWEEN_WHEELS)
    RobotValues.set_robot_height(settings.Robot.HEIGHT)

    backend = Simulator(settings, individual)
    for _ in range(math.ceil(settings.Simulation.TIME_LENGTH / settings.Simulation.TIME_STEP)):
        backend.step()

    return backend.total_score()


def main():
    settings = MySettings()

    ic("=" * 50)
    ic("OPTIMIZATION CLIENT")
    ic("=" * 50)
    ic(f"Server: {settings.Server.HOST}:{settings.Server.PORT}")
    ic("-" * 30)
    ic("Connecting to server...")
    ic("Press Ctrl+C to disconnect")
    ic("=" * 50)

    connect_to_server(
        settings.Server.HOST,
        settings.Server.PORT,
        evaluation_function=evaluation_function,
    )


if __name__ == "__main__":
    main()
