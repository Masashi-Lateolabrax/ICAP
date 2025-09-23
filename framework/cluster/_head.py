import asyncio
from functools import partial
from typing import Optional

from ._network import (
    StateContent,
    AsyncTaskPacket, WorkerPacket, WorkerPacketType,
    ManagedTunnel, AsyncTaskTunnelChild
)
from ._utils import relay_routine


async def head_routine(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter, tunnel: AsyncTaskTunnelChild, timeout: float
):
    while relay_routine(reader, writer, tunnel, timeout):
        pass


class Head:
    def __init__(self):
        self.tunnel = ManagedTunnel()
        self.server = None

    async def start(self, address, port, timeout: float):
        if self.server is not None:
            raise RuntimeError("Server is already running")

        def body_fn():
            child = self.tunnel.spawn_child()
            return partial(head_routine, tunnel=child, timeout=timeout)

        self.server = await asyncio.start_server(body_fn(), address, port)

    async def stop(self):
        for id_ in self.tunnel.get_ids():
            await self.tunnel.send(id_, AsyncTaskPacket.stop_packet())

    async def cleanup(self, timeout: float = 5):
        await self.tunnel.cleanup(timeout)

    async def get_ids(self):
        return self.tunnel.get_ids()

    async def send(self, id_, packet: WorkerPacket):
        packet = AsyncTaskPacket.worker_packet(packet)
        await self.tunnel.send(id_, packet)

    async def receive(self, id_) -> Optional[WorkerPacket]:
        return await self.tunnel.receive(id_)

    async def request_worker_load(self, id_, num_payloads: int):
        await self.send(
            id_,
            WorkerPacket.request_load(num_payloads)
        )

    async def get_worker_load(self, id_) -> Optional[float]:
        response = await self.tunnel.receive(id_, WorkerPacketType.LOAD)
        if response is None:
            return None
        if not isinstance(response.content, float):
            raise ValueError("Invalid response type. Expected float.")
        return response.content

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
