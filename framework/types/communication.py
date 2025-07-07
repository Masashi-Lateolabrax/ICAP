from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass


class PacketType(Enum):
    """
    Packet types for client-server communication.
    
    HANDSHAKE: Initial connection setup - no data
    HEARTBEAT: Regular keepalive with processing speed data
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
