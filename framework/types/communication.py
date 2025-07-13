from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass
import socket
import time

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
        self.calculation_start_time: Optional[float] = None
        self.calculation_end_time: Optional[float] = None

    @property
    def throughput(self) -> float:
        if self.calculation_start_time is None or self.calculation_end_time is None:
            return float('nan')
        duration = self.calculation_end_time - self.calculation_start_time
        if duration <= 0:
            return float('nan')
        return ic(len(self.assigned_individuals) / duration)
