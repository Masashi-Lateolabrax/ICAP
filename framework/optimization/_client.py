import socket
import logging
from typing import Optional

from ..prelude import *
from ._connection_utils import send_individuals, receive_individuals


class OptimizationClient:
    def __init__(self, host: str = 'localhost', port: int = 5000):
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None

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

    def run_evaluation_loop(self, evaluation_function: EvaluationFunction) -> None:
        if self._socket is not None:
            raise RuntimeError("Client is already connected.")

        if not self.connect():
            raise RuntimeError("Failed to connect to server.")

        try:
            assert self._socket is not None  # connect() succeeded
            while True:
                success, individuals = receive_individuals(self._socket)
                if not success:
                    logging.error("Fatal error receiving data from server, disconnecting")
                    self.disconnect()
                    break

                if individuals is None:
                    logging.info("Connection is healthy but no data received.")
                    self.disconnect()
                    break

                elif not isinstance(individuals, list):
                    logging.error(f"Received invalid data type: {type(individuals)}")
                    self.disconnect()
                    break

                elif len(individuals) == 0:
                    logging.info("Received empty list of individuals, continuing...")
                    # TODO: NEED SLEEP HERE! This sleep improves performance by reducing CPU usage and prevents disconnecting
                    continue

                logging.debug(f"Received {len(individuals)} individuals")
                for individual in individuals:
                    try:
                        fitness = evaluation_function(individual)
                        individual.set_fitness(fitness)
                        logging.debug(f"Evaluated individual with fitness: {fitness}")
                    except Exception as e:
                        logging.error(f"Error during evaluation function execution: {e}")
                        individual.set_fitness(float('inf'))

                if not send_individuals(self._socket, individuals):
                    logging.error("Failed to send result to server")
                    self.disconnect()
                    break

        except KeyboardInterrupt:
            logging.info("Client shutting down...")

        finally:
            self.disconnect()


def connect_to_server(server_address: str, port: int, evaluation_function: EvaluationFunction) -> None:
    client = OptimizationClient(server_address, port)
    client.run_evaluation_loop(evaluation_function)
