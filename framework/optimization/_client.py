import socket
import logging
import threading
from typing import Optional, Callable
import time
import os

from ..prelude import *
from ._connection_utils import send_individuals, receive_individuals


class OptimizationClient:
    def __init__(self, host: str = 'localhost', port: int = 5000):
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._stop_event = threading.Event()

    def connect(self) -> bool:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(10.0)
            self._socket.connect((self.host, self.port))
            print(f"Connected to server: {self.host}:{self.port}")
            return True
        except (socket.error, socket.timeout) as e:
            logging.error(f"Failed to connect to server: {e}")
            return False

    def disconnect(self) -> None:
        if self._socket:
            try:
                self._socket.close()
            except Exception as e:
                logging.error(f"Error closing socket: {e}")
            self._socket = None

    def run_evaluation_loop(self, evaluation_function: EvaluationFunction, handler: Optional[Callable] = None) -> None:
        if self._socket is not None:
            raise RuntimeError("Client is already connected.")

        if not self.connect():
            raise RuntimeError("Failed to connect to server.")

        try:
            assert self._socket is not None  # connect() succeeded
            while not self._stop_event.is_set():
                success, individuals = receive_individuals(self._socket)
                if not success:
                    logging.error("Fatal error receiving data from server, disconnecting")
                    self.disconnect()
                    break

                if individuals is None:
                    logging.info("Connection is healthy but no data received.")
                    self.disconnect()
                    break

                if not isinstance(individuals, list):
                    logging.error(f"Received invalid data type: {type(individuals)}")
                    self.disconnect()
                    break

                if len(individuals) == 0:
                    logging.info("Received empty list of individuals, continuing...")
                    print(f"[{self._socket.getsockname()[1]}] Received empty list of individuals, continuing...")
                    if self._stop_event.wait(0.01):
                        break
                    continue

                logging.debug(f"Received {len(individuals)} individuals")
                ave_fitness = 0
                start_time = time.time()
                for individual in individuals:
                    try:
                        individual.timer_start()
                        fitness = evaluation_function(individual)
                        individual.timer_end()

                        ave_fitness += fitness
                        individual.set_fitness(fitness)
                        individual.set_calculation_state(CalculationState.FINISHED)

                        logging.debug(f"Evaluated individual with fitness: {fitness}")
                    except Exception as e:
                        logging.error(f"Error during evaluation function execution: {e}")
                        individual.set_fitness(float('inf'))

                ave_fitness /= len(individuals)
                speed = len(individuals) / (max(1e-6, time.time() - start_time))

                metrics = ProcessMetrics(
                    process_id=os.getpid(),
                    num_individuals=len(individuals),
                    speed=speed,
                    average_fitness=ave_fitness,
                    timestamp=time.time()
                )
                logging.info(metrics.format_log_message())

                if handler:
                    handler(metrics)
                else:
                    print(metrics.format_log_message())

                if not send_individuals(self._socket, individuals):
                    logging.error("Failed to send result to server")
                    self.disconnect()
                    break

        except KeyboardInterrupt:
            logging.info("Client shutting down...")
            self._stop_event.set()

        finally:
            self.disconnect()


def connect_to_server(
        server_address: str,
        port: int,
        evaluation_function: EvaluationFunction,
        handler: Optional[Callable] = None
) -> None:
    client = OptimizationClient(server_address, port)
    client.run_evaluation_loop(evaluation_function, handler)
