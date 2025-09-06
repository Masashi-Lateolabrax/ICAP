import dataclasses
import logging
from enum import Enum
from typing import Any, Optional, Self
from dataclasses import dataclass
import socket
import time
import hashlib
import datetime

import numpy as np
from icecream import ic

from .optimization import Individual


class TaskProgress(Enum):
    WAITING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3

    def __str__(self) -> str:
        return self.name



@dataclasses.dataclass(frozen=True)
class Task:
    hash: bytes
    progress: TaskProgress
    timestamp: datetime.datetime

    settings: Any
    parameter: np.ndarray
    result: Optional[float]

    rng_seed: int

    @classmethod
    def new(cls, settings, parameter: np.ndarray, rng_seed: int) -> Self:
        return cls(
            hash=hashlib.md5(
                parameter.tobytes() + str(parameter.shape).encode() + str(parameter.dtype).encode()
            ).digest(),
            progress=TaskProgress.WAITING,
            timestamp=datetime.datetime.now(datetime.UTC),

            settings=settings,
            parameter=parameter,
            result=None,

            rng_seed=rng_seed
        )

    def replace(
            self,
            progress: Optional[TaskProgress] = None,

            settings: Optional[Any] = None,
            parameter: Optional[np.ndarray] = None,
            result: Optional[float] = None,

            rng_seed: Optional[int] = None
    ) -> Self:
        if parameter is None:
            hash_ = self.hash
            parameter = self.parameter

        else:
            hash_ = hashlib.md5(
                parameter.tobytes() + str(parameter.shape).encode() + str(parameter.dtype).encode()
            ).digest()

        return dataclasses.replace(
            self,
            hash=hash_,
            progress=self.progress if progress is None else progress,
            timestamp=datetime.datetime.now(datetime.UTC),

            settings=self.settings if settings is None else settings,
            parameter=parameter,
            result=self.result if result is None else result,

            rng_seed=self.rng_seed if rng_seed is None else rng_seed
        )

    def is_completed(self):
        return self.progress == TaskProgress.COMPLETED

    def is_waiting(self):
        return self.progress == TaskProgress.WAITING

    def is_running(self):
        return self.progress == TaskProgress.RUNNING


class ClientStatistics:
    performance: float


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
        self.throughput = 0.8 * throughput + 0.2 * ic(len(self.assigned_individuals) / duration)
