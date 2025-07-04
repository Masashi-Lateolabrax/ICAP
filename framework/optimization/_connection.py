import socket
import logging
from typing import Optional

from ..prelude import *
from ._connection_utils import send_individual, receive_individual


class Connection:
    def __init__(self, socket_: socket.socket):
        self._socket = socket_
        self._socket.settimeout(1.0)
        self._assigned_individual: Optional[Individual] = None

    @property
    def has_assigned_individual(self) -> bool:
        return self._assigned_individual is not None

    @property
    def is_alive(self) -> bool:
        return self._socket is not None

    def assign(self, individual: Individual) -> None:
        if self._assigned_individual is not None:
            logging.error("Attempted to assign individual to connection that already has one")
            raise ValueError("Individual is already assigned to this connection.")
        if not isinstance(individual, Individual):
            logging.error(f"Attempted to assign non-Individual object: {type(individual)}")
            raise TypeError("Can only assign Individual objects")
        self._assigned_individual = individual
        self._assigned_individual.set_calculation_state(CalculationState.NOT_STARTED)
        logging.debug(f"Assigned individual with shape {individual.shape} to connection")

    def update(self) -> Optional[float]:
        if self._assigned_individual is None:
            logging.warning("Connection update called with no assigned individual")
            return None

        logging.debug(f"Updating connection with individual state: {self._assigned_individual.get_calculation_state()}")

        success = send_individual(self._socket, self._assigned_individual)
        if not success:
            logging.error("Connection closed while sending individual")
            self.close_by_fatal_error()
            return None

        self._assigned_individual.set_calculation_state(CalculationState.CALCULATING)
        logging.debug("Individual sent to client, state set to CALCULATING")

        success, individual = receive_individual(self._socket)
        if not success:
            logging.error("Connection closed while receiving individual")
            self.close_by_fatal_error()
            return None

        if individual is None:
            logging.error("Received None individual from client")
            return None

        self._assigned_individual.copy_from(individual)
        self._assigned_individual.set_calculation_state(
            CalculationState.FINISHED
        )
        fitness = individual.get_fitness()
        logging.debug(f"Received individual with fitness: {fitness}")
        self._assigned_individual = None
        return fitness

    def close_by_fatal_error(self) -> None:
        logging.info("Closing connection due to fatal error")
        print("CONNECTION DISCONNECTED: Connection closed due to fatal error")
        if self._assigned_individual is not None:
            if not self._assigned_individual.is_corrupted:
                self._assigned_individual.set_calculation_state(
                    CalculationState.CORRUPTED
                )
                logging.debug(f"Marked individual as corrupted (fitness: {self._assigned_individual.get_fitness()})")
            self._assigned_individual = None
        if self._socket is not None:
            try:
                self._socket.close()
                logging.debug("Socket closed successfully")
            except Exception as e:
                logging.error(f"Error closing socket: {e}")
            self._socket = None
