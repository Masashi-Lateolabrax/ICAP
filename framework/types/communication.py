import logging
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass
import socket
import time

import numpy as np
from icecream import ic

from .optimization import Individual


class CommunicationResult(Enum):
    SUCCESS = 0
    OVER_ATTEMPT_COUNT = 1
    CONNECTION_ERROR = 2
    DISCONNECTED = 3
    BROKEN_DATA = 4
    TIMEOUT = 5


class PacketType(Enum):
    """
    Packet types for client-server communication.
    
    HANDSHAKE: Initial connection setup - no data
    HEARTBEAT: Regular keepalive signal - no data
    REQUEST: Request for Individuals from server - no data
    RESPONSE: Send Individuals to server - contains Individual data
    DISCONNECTION: Notify before disconnecting - no data
    ACK: Acknowledgment response - may contain data or be empty
    """
    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    REQUEST = "request"
    RESPONSE = "response"
    DISCONNECTION = "disconnection"
    ACK = "acknowledgment"


@dataclass
class Packet:
    _packet_type: Optional[PacketType] = None
    data: Optional[Any] = None

    @property
    def packet_type(self) -> Optional[PacketType]:
        return self._packet_type


class SocketState:
    def __init__(self, sock: socket.socket):
        peer = sock.getpeername()
        self.address = f"{peer[0]}:{peer[1]}"
        self.last_heartbeat = time.time()
        self.assigned_individuals: Optional[list[Individual]] = None
        self.__timer: float = -1
        self.throughput: float = float('nan')

    def start_timer(self, current_time: Optional[float] = None):
        if current_time is None:
            current_time = time.time()
        self.__timer = current_time

    def stop_timer(self, current_time: Optional[float] = None):
        if self.assigned_individuals is None:
            logging.error("No assigned individuals")
            return
        if self.__timer < 0:
            logging.error("Timer was not started")
            return

        if current_time is None:
            current_time = time.time()
        duration = current_time - self.__timer

        if duration <= 0:
            logging.error("Invalid duration: %s", duration)
            return

        throughput = self.throughput if not np.isnan(self.throughput) else 0.0
        self.throughput = 0.2 * throughput + 0.8 * ic(len(self.assigned_individuals) / duration)
