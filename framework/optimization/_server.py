import queue
import socket
import threading
import logging
import time
from datetime import datetime
from queue import Queue

from ._connection import Connection
from ._cmaes import CMAES
from ._distribution import Distribution
from ..prelude import *


class Reporter:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.buffer: list[str] = []
        self.last_output_time = time.time()

    def add(self, address: str, fitness_values: list[float], throughput: float) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")

        average_fitness = sum(fitness_values) / len(fitness_values)
        variance_fitness = sum((f - average_fitness) ** 2 for f in fitness_values) / len(fitness_values)
        error_count = sum(1 for f in fitness_values if f == float("inf"))

        self.buffer.append(
            f"[{timestamp}] [{address}] AveFitness:{average_fitness}, Variance:{variance_fitness}, Error:{error_count}, Throughput:{throughput:.1f} ind/sec"
        )

    def should_output(self) -> bool:
        current_time = time.time()
        return current_time - self.last_output_time >= self.interval

    def output(self) -> None:
        if self.buffer:
            for message in self.buffer:
                print(message)
            self.buffer.clear()
            self.last_output_time = time.time()


def server_thread(settings: Settings, conn_queue, stop_event):
    cmaes = CMAES(
        dimension=settings.Optimization.dimension,
        mean=None,
        sigma=settings.Optimization.sigma,
        population_size=settings.Optimization.population_size,
    )
    distribution = Distribution(cmaes)
    connections: list[Connection] = []
    reporter = Reporter(1.0)

    while not stop_event.is_set():
        while not conn_queue.empty():
            try:
                conn = conn_queue.get()
                if isinstance(conn, Connection):
                    connections.append(conn)
            except queue.Empty:
                break

        if len(connections) == 0:
            if stop_event.is_set():
                break
            logging.debug("No connections available, waiting for clients...")
            print("DEBUG: No connections available, waiting for clients...")
            threading.Event().wait(1)
            continue

        for i in range(len(connections) - 1, -1, -1):
            conn = connections[i]

            try:
                if not conn.is_healthy:
                    conn.close_gracefully()
                    connections.pop(i)
                    continue

                if not conn.has_assigned_individuals:
                    batch_size = distribution.get_batch_size(conn)
                    batch = cmaes.get_individuals(batch_size)
                    if batch:
                        conn.assign_individuals(batch)
                        logging.debug(f"Assigned batch of {len(batch)} individuals to connection {i}")
                    else:
                        logging.debug(f"No individuals available to assign to connection {i}")
                        continue

                fitness_values, throughput = conn.update()
                if fitness_values is not None:
                    reporter.add(
                        address=conn.address,
                        fitness_values=fitness_values,
                        throughput=throughput
                    )
                    distribution.register_performance(conn)

            except Exception as e:
                logging.error(f"Error handling connection {i}: {e}")
                conn.close_gracefully()
                connections.pop(i)

        if cmaes.ready_to_update():
            try:
                distribution.update()  # The distribution update don't depend on the cmaes state.
                solutions = cmaes.update()

                if cmaes.should_stop():
                    logging.info("Optimization convergence reached, stopping server")
                    stop_event.set()

                    best_solution = None
                    for solution in solutions:
                        if best_solution is None or solution[1] < best_solution[1]:
                            best_solution = solution

                    if best_solution:
                        logging.info(f"Best solution found: {best_solution[0]} with fitness {best_solution[1]}")
                        print(f"Best solution found: {best_solution[0]} with fitness {best_solution[1]}")

            except Exception as e:
                logging.error(f"Error updating CMAES: {e}")

        if reporter.should_output():
            reporter.output()

    reporter.output()

    for conn in connections:
        if conn.is_healthy:
            conn.close_gracefully()


class OptimizationServer:
    def __init__(self, settings: Settings):
        self.settings = settings

        self._queue = Queue()
        self.stop_event = threading.Event()
        self.server_thread = None

    def _setup_server_socket(self):
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.settimeout(1.0)
            server_socket.bind((
                self.settings.Server.HOST,
                self.settings.Server.PORT
            ))
            server_socket.listen(self.settings.Server.SOCKET_BACKLOG)
            logging.info(
                f"Server socket bound to {self.settings.Server.HOST}:{self.settings.Server.PORT} with max connections: {self.settings.Server.SOCKET_BACKLOG}")
            return server_socket
        except socket.error as e:
            logging.error(f"Failed to setup server socket: {e}")
            raise RuntimeError(f"Failed to setup server socket: {e}")

    def start_server(self) -> None:
        self.server_thread = threading.Thread(
            target=server_thread,
            args=(self.settings, self._queue, self.stop_event)
        )
        self.server_thread.start()

        server_socket = self._setup_server_socket()

        try:
            logging.info(f"Optimization server listening on {self.settings.Server.HOST}:{self.settings.Server.PORT}")

            while not self.stop_event.is_set():
                try:
                    client_socket, client_address = server_socket.accept()
                    logging.info(f"Client connected: {client_address}")
                    print(f"CONNECTION ESTABLISHED: {client_address}")

                except socket.timeout:
                    threading.Event().wait(4)
                    continue  # Check stop_event and try again

                except socket.error as e:
                    logging.error(f"Error accepting client connection: {e}")
                    continue

                connection = Connection(client_socket)

                try:
                    self._queue.put(connection)
                except Exception as e:
                    logging.error(f"Failed to queue connection: {e}")
                    connection.close_by_fatal_error()


        except KeyboardInterrupt:
            logging.info("Server shutting down...")
        except Exception as e:
            logging.error(f"Server error: {e}")
        finally:
            try:
                server_socket.close()
            except Exception as e:
                logging.error(f"Error closing server socket: {e}")

            self.stop_event.set()
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join()
                print("DEBUG: Server thread joined")
