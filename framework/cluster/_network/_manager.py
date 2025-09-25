import datetime
import logging
import uuid
from typing import Optional

from ...prelude import *
from ._head import Head
from ._worker import Worker


class ConnectionManager:
    def __init__(self, interval: float = 10.0):
        self.last_heartbeat: dict[uuid.UUID, datetime.datetime] = {}
        self.interval = interval

    def _manage_heartbeat(self, id_: uuid.UUID) -> bool:
        if id_ not in self.last_heartbeat:
            self.last_heartbeat[id_] = datetime.datetime.now(tz=datetime.UTC)
            return True
        current = self.last_heartbeat[id_]
        do_send_heartbeat = (current - self.last_heartbeat[id_]).total_seconds() > self.interval
        if do_send_heartbeat:
            self.last_heartbeat[id_] = datetime.datetime.now(tz=datetime.UTC)
        return do_send_heartbeat


class ManagedTunnel:
    def __init__(self):
        self.tunnel = AsyncTaskTunnel()
        self.buffer: dict[uuid.UUID, list[AsyncTaskPacket]] = {}
        self.ping: dict[uuid.UUID, Optional[PingContent]] = {}

    def spawn_child(self) -> AsyncTaskTunnelChild:
        child = self.tunnel.spawn_child()
        self.buffer[child.uuid] = []
        self.ping[child.uuid] = None
        return child

    def get_ids(self) -> list[uuid.UUID]:
        return self.buffer.keys()

    async def _update_buffer(self):
        for id_ in self.tunnel.get_ids():
            while not self.tunnel.empty(id_):
                packet = await self.tunnel.receive(id_, None)
                if packet is None:
                    raise RuntimeError("Here is not reachable")
                self.buffer[id_].append(packet)

    def _take_ping_content_from_buffer(self) -> dict[uuid.UUID, list[PingContent]]:
        result = {}
        for id_ in self.tunnel.get_ids():
            result[id_] = []
            for i in reversed(range(0, len(self.buffer[id_]))):
                packet = self.buffer[id_][i]
                if packet.type != AsyncTaskPacketType.PING:
                    continue
                if not isinstance(packet.content, PingContent):
                    raise ValueError("Invalid ping packet content")
                if packet.content.response_time is None:
                    raise ValueError("Here is not reachable")
                self.buffer[id_].pop(i)
                result[id_].append(packet.content)
        return result

    def _update_ping(self):
        received_pings = self._take_ping_content_from_buffer()
        for id_, pings in received_pings.items():
            if not pings or id_ not in self.ping or self.ping[id_] is None or self.ping[id_].response_time is not None:
                continue
            latest_ping = max(pings, key=lambda p: p.response_time)
            self.ping[id_].response_time = latest_ping.response_time

    async def _check_dead_tunnel(self, timeout: float) -> list[uuid.UUID]:
        await self._update_buffer()
        self._update_ping()
        now = datetime.datetime.now(tz=datetime.UTC)
        dead_ids = []
        for id_, ping in self.ping.items():
            if ping is None or ping.response_time is not None:
                continue
            if (now - ping.create_time).total_seconds() > timeout:
                dead_ids.append(id_)
        return dead_ids

    async def cleanup(self, timeout: float) -> list[uuid.UUID]:
        dead_ids = await self._check_dead_tunnel(timeout)
        for id_ in dead_ids:
            del self.buffer[id_]
            del self.ping[id_]
            self.tunnel.del_id(id_)
        return dead_ids

    def _receive_filtered(
            self, id_: uuid.UUID, expect_worker_type: WorkerPacketType
    ) -> Optional[AsyncTaskPacket]:
        for i in range(0, len(self.buffer[id_])):
            packet = self.buffer[id_][i]

            if packet.type != AsyncTaskPacketType.WORKER_PACKET:
                continue
            if not isinstance(packet.content, WorkerPacket):
                raise ValueError("Invalid response type. Expected WorkerPacket.")

            if packet.content.type == expect_worker_type:
                return self.buffer[id_].pop(i)

        return None

    def _cleanup_buffer(self, id_: uuid.UUID):
        latest_state_packet: dict = {
            "index": None,
            "packet": None
        }

        for i in reversed(range(len(self.buffer[id_]))):
            packet = self.buffer[id_][i]

            if packet.type == AsyncTaskPacketType.WORKER_PACKET and isinstance(packet.content, WorkerPacket):
                logging.warning("Invalid packet content. Expected WorkerPacket.")
                continue

            if packet.content.type == WorkerPacketType.STATE:
                if latest_state_packet["packet"] is None:
                    latest_state_packet["index"] = i
                    latest_state_packet["packet"] = packet
                    continue
                elif latest_state_packet["packet"].timestamp < packet.content.timestamp:
                    latest_state_packet["packet"] = packet
                self.buffer[id_].pop(latest_state_packet["index"])
                latest_state_packet["index"] = i
                continue

            self.buffer[id_].pop(i)

        if latest_state_packet["index"] is not None:
            self.buffer[id_][latest_state_packet["index"]] = latest_state_packet["packet"]

    async def receive(self, id_: uuid.UUID, expect_worker_type: WorkerPacketType = None) -> Optional[WorkerPacket]:
        await self._update_buffer()
        if id_ not in self.buffer:
            raise ValueError("Invalid id")
        if len(self.buffer[id_]) == 0:
            return None
        self._cleanup_buffer(id_)

        if expect_worker_type is None:
            async_packet = self.buffer[id_].pop(0)
        else:
            async_packet = self._receive_filtered(id_, expect_worker_type)

        if async_packet is None:
            return None
        return async_packet.content

    async def send(self, id_: uuid.UUID, packet: WorkerPacket):
        if id_ not in self.buffer:
            raise ValueError("Invalid id")
        packet = AsyncTaskPacket.worker_packet(packet)
        await self.tunnel.send(id_, packet)
