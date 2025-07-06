import socket
import logging
from typing import Optional

from ..prelude import *
from ._connection_utils import send_individuals, receive_individuals


class Connection:
    def __init__(self, socket_: socket.socket):
        self.socket = socket_
        self.socket.settimeout(1.0)
        self._assigned_individuals: Optional[list[Individual]] = None
        self._name = f"{self.socket.getpeername()[0]}:{self.socket.getpeername()[1]}"
        self._throughput = 0

    @property
    def address(self) -> str:
        return f"{self._name}{'(healthy)' if self.is_healthy else '(unhealthy)'}"

    @property
    def has_assigned_individuals(self) -> bool:
        return bool(self._assigned_individuals)

    @property
    def is_healthy(self) -> bool:
        """Check if the connection is healthy by attempting to get socket information"""
        if self.socket is None:
            return False
        try:
            # Try to get peer address - this will fail if connection is broken
            self.socket.getpeername()
            return True
        except (socket.error, OSError):
            return False

    @property
    def throughput(self) -> Optional[float]:
        return self._throughput

    def assign_individuals(self, individuals: list[Individual]) -> None:
        if self.has_assigned_individuals:
            logging.error("Attempted to assign batch to connection that already has one")
            raise ValueError("Batch is already assigned to this connection.")

        if not isinstance(individuals, list) or not all(isinstance(ind, Individual) for ind in individuals):
            logging.error(f"Attempted to assign non-Individual batch: {type(individuals)}")
            raise TypeError("Can only assign list of Individual objects")

        self._assigned_individuals = individuals

        for individual in self._assigned_individuals:
            individual.set_calculation_state(CalculationState.NOT_STARTED)

        logging.debug(f"Assigned batch of {len(individuals)} individuals to connection")

    def send_if_ready(self) -> bool:
        """Send individuals to client if they are ready to be sent.
        
        Returns:
            bool: True if sending was successful or not needed, False if fatal error occurred
        """
        if self._assigned_individuals is None:
            return True  # No individuals to send

        # Only send if individuals are not yet calculating
        if not bool(self._assigned_individuals) or not all([i.is_calculating for i in self._assigned_individuals]):
            batch_size = len(self._assigned_individuals)
            logging.debug(f"Sending batch of {batch_size} individuals to connection")

            success = send_individuals(self.socket, self._assigned_individuals)
            if not success:
                logging.error("Connection closed while sending individual batch")
                self.close_by_fatal_error()
                return False

            for individual in self._assigned_individuals:
                individual.set_calculation_state(CalculationState.CALCULATING)
            logging.debug("Individual batch sent to client, state set to CALCULATING")

        return True

    def receive_if_ready(self) -> Optional[tuple[list[float], float]]:
        """Receive individuals from client if data is available.
        
        Returns:
            tuple[list[float], float]: Fitness values and throughput if successful
            None: If no data available or error occurred
        """
        if self._assigned_individuals is None:
            logging.warning("Connection batch receive called with no assigned batch")
            return None

        # Receiving
        success, individuals = receive_individuals(self.socket, blocking=False)

        if not success:
            logging.error("Connection closed while receiving individual batch")
            self.close_by_fatal_error()
            return None

        if individuals is None:
            logging.info("The calculation in the client is still in progress, no individuals received")
            return None

        if len(individuals) != len(self._assigned_individuals):
            logging.error(
                f"Received batch size {len(individuals)} != sent batch size {len(self._assigned_individuals)}"
            )
            return None

        # Process received individuals
        fitness_values = []
        throughput = 0
        for i, individual in enumerate(individuals):
            self._assigned_individuals[i].copy_from(individual)
            if individual.is_finished:
                fitness = individual.get_fitness()
                fitness_values.append(fitness)
                throughput += 1 / individual.get_elapse()
        throughput /= len(fitness_values)

        self._throughput = 0.8 * self._throughput + 0.2 * throughput

        logging.debug(
            f"Received batch with fitness values: {fitness_values}, throughput: {self.throughput:.2f} individuals/sec"
        )
        self._assigned_individuals = None

        return fitness_values, self.throughput

    def close_by_fatal_error(self) -> None:
        logging.info("Closing connection due to fatal error")
        print("CONNECTION DISCONNECTED: Connection closed due to fatal error")
        self._cleanup_resources()

    def close_gracefully(self) -> None:
        logging.info("Closing connection gracefully")
        print("CONNECTION DISCONNECTED: Connection closed gracefully")
        self._cleanup_resources()

    def _cleanup_resources(self) -> None:
        """Clean up all resources associated with this connection"""
        # Clean up assigned individuals
        if self._assigned_individuals is not None:
            for individual in self._assigned_individuals:
                if not individual.is_corrupted:
                    individual.set_calculation_state(CalculationState.CORRUPTED)
            logging.debug(f"Marked batch of {len(self._assigned_individuals)} individuals as corrupted")
            self._assigned_individuals = None

        # Clean up socket
        if self.socket is not None:
            try:
                # Try to shutdown socket gracefully first
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
                logging.debug("Socket closed successfully")
            except Exception as e:
                logging.error(f"Error closing socket: {e}")
            finally:
                self.socket = None
