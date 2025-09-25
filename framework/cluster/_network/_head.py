import asyncio
import uuid

from icecream import ic

from ...prelude import *
from ._tunnel import Tunnel
from ._utils import relay_routine


async def head_routine(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, tunnel: Tunnel, timeout: float):
    ic(writer.get_extra_info('peername'))
    ic(tunnel.id)
    while await relay_routine(reader, writer, tunnel, timeout):
        pass


class Head:
    def __init__(self):
        self.tunnel: dict[uuid.UUID, Tunnel] = {}
        self.server = None

    async def start(self, address, port, timeout: float):
        if self.server is not None:
            raise RuntimeError("Server is already running")

        async def body_fn(reader, writer):
            parent, child = Tunnel.create()
            self.tunnel[parent.id] = parent
            await head_routine(reader, writer, tunnel=child, timeout=timeout)

        self.server = await asyncio.start_server(body_fn, address, port)

    async def stop(self):
        if self.server is None:
            raise RuntimeError("Server is not running")

        self.server.close()
        await self.server.wait_closed()
        self.server = None

        stop_packet = CoroutinePacket.stop_packet()
        for t in self.tunnel.values():
            await t.send(stop_packet)
        self.tunnel.clear()

    async def delete(self, id_: uuid.UUID):
        await self.tunnel[id_].send(CoroutinePacket.stop_packet())
        del self.tunnel[id_]

    def get_ids(self) -> set[uuid.UUID]:
        return set(self.tunnel.keys())

    def receive(self) -> dict[uuid.UUID, ClusterPacket]:
        received_packets = {}

        for i, t in self.tunnel.items():
            while True:
                packet = t.receive()

                if packet is None:
                    break
                if packet.type != CoroutinePacketType.CLUSTER_PACKET:
                    continue
                if not isinstance(packet.content, ClusterPacket):
                    raise ValueError("Unexpected content type")

                received_packets[i] = packet.content

        return received_packets

    async def send(self, id_: uuid.UUID, packet: ClusterPacket):
        if id_ not in self.tunnel:
            raise ValueError(f"Tunnel with id {id_} does not exist.")
        packet = CoroutinePacket(CoroutinePacketType.CLUSTER_PACKET, packet)
        await self.tunnel[id_].send(packet)
