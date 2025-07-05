import socket
import logging
from typing import Optional

from ..prelude import *
from ._connection_utils import send_individuals, receive_individuals


class Connection:
    def __init__(self, socket_: socket.socket):
        self._socket = socket_
        self._socket.settimeout(1.0)
        self._assigned_individuals: Optional[list[Individual]] = None

    @property
    def address(self) -> str:
        return f"{self._socket.getpeername()[0]}:{self._socket.getpeername()[1]}"

    @property
    def has_assigned_individuals(self) -> bool:
        return self._assigned_individuals is not None

    @property
    def is_alive(self) -> bool:
        return self._socket is not None

    def assign_individuals(self, individuals: list[Individual]) -> None:
        if self._assigned_individuals is not None:
            logging.error("Attempted to assign batch to connection that already has one")
            raise ValueError("Batch is already assigned to this connection.")
        if not isinstance(individuals, list) or not all(isinstance(ind, Individual) for ind in individuals):
            logging.error(f"Attempted to assign non-Individual batch: {type(individuals)}")
            raise TypeError("Can only assign list of Individual objects")
        self._assigned_individuals = individuals
        for individual in self._assigned_individuals:
            individual.set_calculation_state(CalculationState.NOT_STARTED)
        logging.debug(f"Assigned batch of {len(individuals)} individuals to connection")

    def update(self) -> Optional[list[float]]:
        if self._assigned_individuals is None:
            logging.warning("Connection batch update called with no assigned batch")
            return None

        logging.debug(f"Updating connection with batch of {len(self._assigned_individuals)} individuals")

        success = send_individuals(self._socket, self._assigned_individuals)
        if not success:
            logging.error("Connection closed while sending individual batch")
            self.close_by_fatal_error()
            return None

        for individual in self._assigned_individuals:
            individual.set_calculation_state(CalculationState.CALCULATING)
        logging.debug("Individual batch sent to client, state set to CALCULATING")

        success, individuals = receive_individuals(self._socket)
        if not success:
            logging.error("Connection closed while receiving individual batch")
            self.close_by_fatal_error()
            return None

        if individuals is None:
            logging.error("Received None individual batch from client")
            return None

        if len(individuals) != len(self._assigned_individuals):
            logging.error(f"Received batch size {len(individuals)} != sent batch size {len(self._assigned_individuals)}")
            return None

        fitness_values = []
        for i, individual in enumerate(individuals):
            self._assigned_individuals[i].copy_from(individual)
            self._assigned_individuals[i].set_calculation_state(CalculationState.FINISHED)
            fitness = individual.get_fitness()
            fitness_values.append(fitness)

        logging.debug(f"Received batch with fitness values: {fitness_values}")
        self._assigned_individuals = None
        return fitness_values

    def close_by_fatal_error(self) -> None:
        logging.info("Closing connection due to fatal error")
        print("CONNECTION DISCONNECTED: Connection closed due to fatal error")
        if self._assigned_individuals is not None:
            for individual in self._assigned_individuals:
                if not individual.is_corrupted:
                    individual.set_calculation_state(CalculationState.CORRUPTED)
            logging.debug(f"Marked batch of {len(self._assigned_individuals)} individuals as corrupted")
            self._assigned_individuals = None
        if self._socket is not None:
            try:
                self._socket.close()
                logging.debug("Socket closed successfully")
            except Exception as e:
                logging.error(f"Error closing socket: {e}")
            self._socket = None
