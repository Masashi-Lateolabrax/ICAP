from typing import Optional
from datetime import datetime

import numpy as np

from ..prelude import *
from ._network import Worker, ConnectionManager


class Client:
    """Minimal client with periodic state sending and task processing"""

    @classmethod
    async def new(
            cls,
            host: str,
            port: int,
            timeout: float = 1.0,
            heartbeat_interval: float = 5.0,
            heartbeat_timeout: float = 15.0
    ):
        """Create and start a new client"""
        worker = await Worker.start(host, port, timeout)
        client = cls(worker, heartbeat_interval, heartbeat_timeout)
        print(f"SimpleClient connected to {host}:{port}")
        return client

    def __init__(self, worker: Worker, heartbeat_interval: float, heartbeat_timeout: float):
        self.id = worker.id
        self._worker: Optional[Worker] = worker
        self._manager = ConnectionManager(interval=heartbeat_interval, timeout=heartbeat_timeout)

    async def stop(self):
        """Stop the client"""
        if self._worker:
            await self._worker.stop()
            self._worker = None
        print("SimpleClient disconnected")

    async def manage(self):
        if not self._worker:
            return
        dead_ids = self._manager.manage(self._worker)
        if self.id not in dead_ids:
            await self.stop()
        else:
            raise RuntimeError("Here is not reachable")

    def get_task(self) -> Optional[TaskContent]:
        if not self._worker:
            return None

        packet = self._manager.receive(self._worker).get(self.id, None)
        if packet is None:
            return None

        if packet.type != ClusterPacketType.TASK:
            raise ValueError("Unexpected packet type")
        if not isinstance(packet.content, TaskContent):
            raise ValueError("Unexpected content type")

        return packet.content

    async def send_result(self, fitness: list[tuple[np.ndarray, float]], start_time: datetime, end_time: datetime):
        if not self._worker:
            return

        result_content = ResultContent(result=fitness, start_time=start_time, end_time=end_time)
        cluster_packet = ClusterPacket(ClusterPacketType.RESULT, result_content)
        await self._worker.send(cluster_packet)

    async def send_state(self, gpu_usage: float = 0.0, working: bool = False):
        if not self._worker:
            return

        state_content = StateContent(gpu_usage=gpu_usage, working=working)
        cluster_packet = ClusterPacket(ClusterPacketType.STATE, state_content)
        await self._worker.send(cluster_packet)
