"""
Optimization Server

This script starts an optimization server that distributes CMA-ES optimization
tasks to connected clients.
"""

import os
import threading
from datetime import datetime
from icecream import ic
from framework.optimization import OptimizationServer

from controller import RobotNeuralNetwork
from settings import MySettings

# Configure icecream for distributed system debugging
ic.configureOutput(prefix=lambda: f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] SERVER| ')


def main():
    dim = RobotNeuralNetwork().dim

    settings = MySettings()

    settings.Server.HOST = "0.0.0.0"
    settings.Optimization.dimension = dim

    ic("=" * 50)
    ic("OPTIMIZATION SERVER")
    ic("=" * 50)
    ic(f"Host: {settings.Server.HOST}")
    ic(f"Port: {settings.Server.PORT}")
    ic(f"Socket Backlog: {settings.Server.SOCKET_BACKLOG}")
    ic("-" * 30)
    ic(f"Problem dimension: {settings.Optimization.dimension}")
    ic(f"Initial sigma: {settings.Optimization.sigma}")
    ic(f"Population size: {settings.Optimization.population_size}")
    ic("-" * 30)
    ic("Waiting for clients to connect...")
    ic("Press Ctrl+C to stop the server")
    ic("=" * 50)

    server = OptimizationServer(settings)

    try:
        server.start_server()
    except KeyboardInterrupt:
        ic("\n" + "=" * 50)
        ic("Server stopped by user")
        ic("=" * 50)
    except Exception as e:
        ic(f"\nServer error: {e}")


if __name__ == "__main__":
    main()
