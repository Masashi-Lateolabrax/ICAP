import dataclasses
from enum import Enum
from typing import Any, Optional, Self
import hashlib
import datetime

import numpy as np



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
    result: Optional[float]
    progress: TaskProgress
    timestamp: datetime.datetime

    @classmethod
    def new(cls, result: float = None, progress: TaskProgress = TaskProgress.WAITING) -> Self:
        return cls(
            result=result,
            progress=progress,
            timestamp=datetime.datetime.now(datetime.UTC)
        )

    def hash(self) -> bytes:
        result_hash = hashlib.md5(str(self.result).encode())
        progress_bytes = str(self.progress.value).encode()
        timestamp_bytes = str(self.timestamp).encode()
        return hashlib.md5(result_hash + progress_bytes + timestamp_bytes).digest()

    def replace(
            self,
            result: Optional[float] = None,
            progress: Optional[TaskProgress] = None,
            update_timestamp: bool = True,
    ) -> Self:
        if result is None and progress is None and not update_timestamp:
            return self

        update_timestamp = update_timestamp or result is not None or progress is not None

        result = self.result if result is None else result
        progress = self.progress if progress is None else progress
        timestamp = self.timestamp if not update_timestamp else datetime.datetime.now(datetime.UTC)
        return dataclasses.replace(
            self,
            result=result,
            progress=progress,
            timestamp=timestamp
        )


@dataclasses.dataclass(frozen=True)
class TaskContent:
    settings: Any
    parameter: np.ndarray
    rng_seed: int

    def hash(self) -> bytes:
        parameter_bytes = (
                self.parameter.tobytes() + str(self.parameter.shape).encode() + str(self.parameter.dtype).encode()
        )
        settings_bytes = str(self.settings).encode()
        rng_seed_bytes = str(self.rng_seed).encode()
        return hashlib.md5(parameter_bytes + settings_bytes + rng_seed_bytes).digest()

    def replace(
            self,
            settings: Optional[Any] = None,
            parameter: Optional[np.ndarray] = None,
            rng_seed: Optional[int] = None
    ) -> Self:
        return dataclasses.replace(
            self,
            settings=settings if settings is not None else self.settings,
            parameter=parameter if parameter is not None else self.parameter,
            rng_seed=rng_seed if rng_seed is not None else self.rng_seed
        )


@dataclasses.dataclass(frozen=True)
class TaskID:
    content_hash: bytes
    state_hash: bytes
    fingerprint: bytes

    @classmethod
    def new(
            cls,
            content: TaskContent,
            state: TaskState,
    ) -> Self:
        content_hash = content.hash()
        state_hash = state.hash()
        return cls(
            content_hash=content_hash,
            state_hash=state_hash,
            fingerprint=hashlib.md5(content_hash + state_hash).digest()
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
        fingerprint = self.fingerprint
        if content is not None:
            content_hash = content.hash()
            if self.content_hash != content_hash:
                fingerprint = None

        if state is not None:
            state_hash = state.hash()
            if self.state_hash != state_hash:
                fingerprint = None

        if fingerprint is None:
            fingerprint = hashlib.md5(content_hash + state_hash).digest()

        return dataclasses.replace(
            self,
            content_hash=content_hash,
            state_hash=state_hash,
            fingerprint=fingerprint
        )


@dataclasses.dataclass(frozen=True)
class Task:
    id: TaskID
    content: TaskContent
    state: TaskState

    @property
    def fingerprint(self) -> bytes:
        return self.id.fingerprint

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
        return self.state.result

    @property
    def rng_seed(self) -> int:
        return self.content.rng_seed

    @classmethod
    def new(cls, settings, parameter: np.ndarray, rng_seed: int) -> Self:
        content = TaskContent(
            settings=settings,
            parameter=parameter,
            rng_seed=rng_seed
        )
        state = TaskState.new(
            result=None,
            progress=TaskProgress.WAITING
        )
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
            rng_seed=rng_seed
        )
        new_state = self.state.replace(
            result=result,
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
