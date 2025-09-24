import asyncio
import uuid
from functools import partial
from typing import Optional

from ._network import (
    StateContent, TaskContent, ResultContent,
    AsyncTaskPacket, WorkerPacket, WorkerPacketType,
    ManagedTunnel, AsyncTaskTunnelChild
)
from ._utils import relay_routine


async def head_routine(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter, tunnel: AsyncTaskTunnelChild, timeout: float
):
    while await relay_routine(reader, writer, tunnel, timeout):
        pass


class Head:
    def __init__(self):
        self.tunnel = ManagedTunnel()
        self.server = None

    async def start(self, address, port, timeout: float):
        if self.server is not None:
            raise RuntimeError("Server is already running")

        async def body_fn(reader, writer):
            child = self.tunnel.spawn_child()
            await head_routine(reader, writer, tunnel=child, timeout=timeout)

        self.server = await asyncio.start_server(body_fn, address, port)

    async def stop(self):
        for id_ in self.tunnel.get_ids():
            await self.tunnel.send(id_, AsyncTaskPacket.stop_packet())

    async def cleanup(self, timeout: float = 5) -> list[uuid.UUID]:
        return await self.tunnel.cleanup(timeout)

    async def get_ids(self):
        return self.tunnel.get_ids()

    async def send(self, id_, packet: WorkerPacket):
        packet = AsyncTaskPacket.worker_packet(packet)
        await self.tunnel.send(id_, packet)

    async def receive(self, id_) -> Optional[WorkerPacket]:
        return await self.tunnel.receive(id_)

    async def request_worker_state(self, id_):
        await self.send(
            id_,
            WorkerPacket.request_state()
        )

    async def get_worker_state(self, id_) -> Optional[StateContent]:
        response = await self.tunnel.receive(id_, WorkerPacketType.STATE)
        if response is None:
            return None
        if not isinstance(response.content, StateContent):
            raise ValueError("Invalid response type. Expected StateContent.")
        return response.content

    async def send_worker_task(self, id_: uuid.UUID, task: TaskContent):
        await self.send(
            id_,
            WorkerPacket(WorkerPacketType.TASK, task)
        )

    async def get_worker_result(self, id_: uuid.UUID) -> Optional[ResultContent]:
        response = await self.tunnel.receive(id_, WorkerPacketType.RESULT)
        if response is None:
            return None
        if not isinstance(response.content, ResultContent):
            raise ValueError("Invalid response type. Expected TaskContent.")
        return response.content
