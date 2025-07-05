"""
Optimization Server

This script starts an optimization server that distributes CMA-ES optimization
tasks to connected clients.
"""

from framework.optimization import OptimizationServer
from framework.prelude import *


def main():
    settings = Settings()

    settings.Server.HOST = "0.0.0.0"

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

    server = OptimizationServer(settings)

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
