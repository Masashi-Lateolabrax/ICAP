import socket
import threading
import logging
import time
import select
from datetime import datetime
from queue import Queue
import math

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
        if not fitness_values:
            logging.warning(f"No fitness values provided for address {address}")
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        average_fitness = sum(fitness_values) / len(fitness_values)
        variance_fitness = math.sqrt(sum((f - average_fitness) ** 2 for f in fitness_values) / len(fitness_values))
        error_count = sum(1 for f in fitness_values if f == float("inf"))

        message = (f"[{timestamp}] [{address}] "
                   f"Num:{len(fitness_values)}, "
                   f"AveFitness:{average_fitness:.2f}, "
                   f"Variance:{variance_fitness:.2f}, "
                   f"Error:{error_count}, "
                   f"Throughput:{throughput:.1f} ind/sec")
        self.buffer.append(message)

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
    distribution = Distribution()
    connections: set[Connection] = set()
    reporter = Reporter(1.0)

    while not stop_event.is_set():
        while not conn_queue.empty():
            conn = conn_queue.get()
            if isinstance(conn, Connection):
                connections.add(conn)
                distribution.add_new_connection(conn)

        if len(connections) == 0:
            if stop_event.is_set():
                break
            logging.debug("No connections available, waiting for clients...")
            print("DEBUG: No connections available, waiting for clients...")
            time.sleep(1)
            continue

        # Clean up unhealthy connections and assign individuals
        unhealthy_connections = set()
        assigned_connections = {}

        for conn in connections:
            if not conn.is_healthy:
                conn.close_gracefully()
                unhealthy_connections.add(conn)
                continue

            if not conn.has_assigned_individuals:
                batch_size = distribution.get_batch_size(conn)
                batch = cmaes.get_individuals(batch_size)
                if batch is not None:
                    conn.assign_individuals(batch)
                    logging.debug(f"Assigned batch of {len(batch)} individuals to connection {conn.address}")
                else:
                    logging.debug(f"No individuals available to assign to connection {conn.address}")
            else:
                assigned_connections[conn.socket] = conn

        connections -= unhealthy_connections

        ready_sockets = []
        if assigned_connections:
            try:
                sockets = list(assigned_connections.keys())
                ready_sockets, _, _ = select.select(sockets, [], sockets, 0.1)
            except select.error as e:
                logging.error(f"Select error: {e}")

        # Send individuals to all connections that need them
        for conn in connections:
            if conn.socket in ready_sockets:
                try:
                    reply = conn.receive_if_ready()
                    if reply is not None:
                        fitness_values, throughput = reply
                        reporter.add(
                            address=conn.address,
                            fitness_values=fitness_values,
                            throughput=throughput
                        )
                        distribution.register_throughput(conn, throughput)

                    continue

                except Exception as e:
                    logging.error(f"Error handling connection {conn.address}: {e}")
                    conn.close_gracefully()
                    connections.discard(conn)

            if conn.has_assigned_individuals and conn.is_healthy:
                try:
                    if not conn.send_if_ready():
                        conn.close_gracefully()
                        connections.discard(conn)

                except Exception as e:
                    logging.error(f"Error sending to connection {conn.address}: {e}")
                    conn.close_gracefully()
                    connections.discard(conn)

        if cmaes.ready_to_update():
            try:
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

        distribution.update(cmaes.num_ready_individuals, connections)

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
                    time.sleep(4)
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
