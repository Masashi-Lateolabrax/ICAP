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

    def is_waiting(self) -> bool:
        return self == TaskProgress.WAITING

    def is_running(self) -> bool:
        return self == TaskProgress.RUNNING

    def is_completed(self) -> bool:
        return self == TaskProgress.COMPLETED

    def is_failed(self) -> bool:
        return self == TaskProgress.FAILED


@dataclasses.dataclass(frozen=True)
class TaskState:
    progress: TaskProgress
    timestamp: datetime.datetime

    @classmethod
    def new(cls, progress: TaskProgress = TaskProgress.WAITING) -> Self:
        return cls(
            progress=progress,
            timestamp=datetime.datetime.now(datetime.UTC)
        )

    def hash(self) -> bytes:
        progress_bytes = str(self.progress.value).encode()
        timestamp_bytes = str(self.timestamp).encode()
        return hashlib.md5(progress_bytes + timestamp_bytes).digest()

    def replace(
            self,
            progress: Optional[TaskProgress] = None,
            update_timestamp: bool = True,
    ) -> Self:
        if progress is None and not update_timestamp:
            return self
        elif progress == self.progress and not update_timestamp:
            return self
        return dataclasses.replace(
            self,
            progress=self.progress if progress is None else progress,
            timestamp=datetime.datetime.now(datetime.UTC)
        )


@dataclasses.dataclass(frozen=True)
class TaskContent:
    settings: Any
    parameter: np.ndarray
    result: Optional[float]
    rng_seed: int

    def hash(self) -> bytes:
        parameter_bytes = (
                self.parameter.tobytes() + str(self.parameter.shape).encode() + str(self.parameter.dtype).encode()
        )
        result_bytes = str(self.result).encode()
        settings_bytes = str(self.settings).encode()
        rng_seed_bytes = str(self.rng_seed).encode()
        return hashlib.md5(parameter_bytes + result_bytes + settings_bytes + rng_seed_bytes).digest()

    def replace(
            self,
            settings: Optional[Any] = None,
            parameter: Optional[np.ndarray] = None,
            result: Optional[float] = None,
            rng_seed: Optional[int] = None
    ) -> Self:
        return dataclasses.replace(
            self,
            settings=settings if settings is not None else self.settings,
            parameter=parameter if parameter is not None else self.parameter,
            result=result if result is not None else self.result,
            rng_seed=rng_seed if rng_seed is not None else self.rng_seed
        )


@dataclasses.dataclass(frozen=True)
class TaskID:
    content_hash: bytes
    state_hash: bytes
    id: bytes

    @classmethod
    def new(
            cls,
            content: TaskContent,
            state: TaskState,
    ) -> Self:
        return cls(
            content_hash=content.hash(),
            state_hash=state.hash(),
            id=hashlib.md5(content.hash() + state.hash()).digest()
        )

    def replace(
            self,
            content: Optional[TaskContent] = None,
            state: Optional[TaskState] = None
    ) -> Self:
        if content is None and state is None:
            return self

        content_hash = self.content_hash
        state_hash = self.state_hash
        id_ = self.id
        if content is not None:
            content_hash = content.hash()
            if self.content_hash != content_hash:
                id_ = None

        if state is not None:
            state_hash = state.hash()
            if self.state_hash != state_hash:
                id_ = None

        if id_ is None:
            id_ = hashlib.md5(content_hash + state_hash).digest()

        return dataclasses.replace(
            self,
            content_hash=content_hash,
            state_hash=state_hash,
            id=id_
        )

    def __eq__(self, other):
        if not isinstance(other, TaskID):
            return NotImplemented
        return self.id == other.id


@dataclasses.dataclass(frozen=True)
class Task:
    id: TaskID
    content: TaskContent
    state: TaskState

    @property
    def hash(self) -> bytes:
        return self.id.id

    @property
    def progress(self) -> TaskProgress:
        return self.state.progress

    @property
    def timestamp(self) -> datetime.datetime:
        return self.state.timestamp

    @property
    def settings(self) -> Any:
        return self.content.settings

    @property
    def parameter(self) -> np.ndarray:
        return self.content.parameter

    @property
    def result(self) -> Optional[float]:
        return self.content.result

    @property
    def rng_seed(self) -> int:
        return self.content.rng_seed

    @classmethod
    def new(cls, settings, parameter: np.ndarray, rng_seed: int) -> Self:
        content = TaskContent(
            settings=settings,
            parameter=parameter,
            result=None,
            rng_seed=rng_seed
        )
        state = TaskState.new(progress=TaskProgress.WAITING)
        return cls(
            id=TaskID.new(content, state),
            content=content,
            state=state
        )

    def replace(
            self,
            progress: Optional[TaskProgress] = None,

            settings: Optional[Any] = None,
            parameter: Optional[np.ndarray] = None,
            result: Optional[float] = None,

            rng_seed: Optional[int] = None
    ) -> Self:
        new_content = self.content.replace(
            settings=settings,
            parameter=parameter,
            result=result,
            rng_seed=rng_seed
        )
        new_state = self.state.replace(
            progress=progress,
            update_timestamp=new_content.hash() != self.id.content_hash
        )
        new_id = self.id.replace(
            content=new_content,
            state=new_state
        )

        return dataclasses.replace(
            self,
            content=new_content,
            state=new_state,
            id=new_id
        )


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
