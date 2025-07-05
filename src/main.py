import math

import numpy as np

from framework.prelude import *
from framework.optimization import connect_to_server

from settings import MySettings
from simulator import Simulator


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
    settings = Settings()

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
    )


if __name__ == "__main__":
    main()
