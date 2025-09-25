import datetime
import logging
import uuid
from typing import Optional

from ...prelude import *
from ._head import Head
from ._worker import Worker


class ConnectionManager:
    def __init__(self, interval: float = 10.0, timeout: float = 30.0):
        self.last_heartbeat: dict[uuid.UUID, datetime.datetime] = {}
        self.interval = interval
        self.timeout = timeout

    def _manage_heartbeat(self, id_: uuid.UUID) -> bool:
        if id_ not in self.last_heartbeat:
            self.last_heartbeat[id_] = datetime.datetime.now(tz=datetime.UTC)
            return True
        current = self.last_heartbeat[id_]
        do_send_heartbeat = (current - self.last_heartbeat[id_]).total_seconds() > self.interval
        if do_send_heartbeat:
            self.last_heartbeat[id_] = datetime.datetime.now(tz=datetime.UTC)
        return do_send_heartbeat

    def _manage_head(self, connection: Head):
        for i in connection.get_ids():
            if self._manage_heartbeat(i):
                heartbeat = HeartbeatContent()
                packet = ClusterPacket(ClusterPacketType.HEARTBEAT, heartbeat)
                packet = CoroutinePacket(CoroutinePacketType.CLUSTER_PACKET, packet)
                connection.send(i, packet)

    def _manage_worker(self, connection: Worker):
        id_ = connection.id
        if self._manage_heartbeat(id_):
            heartbeat = HeartbeatContent()
            packet = ClusterPacket(ClusterPacketType.HEARTBEAT, heartbeat)
            packet = CoroutinePacket(CoroutinePacketType.CLUSTER_PACKET, packet)
            connection.send(packet)

    def _manage_dead(self) -> list[uuid.UUID]:
        dead_ids = []
        current = datetime.datetime.now(tz=datetime.UTC)
        for id_, last in list(self.last_heartbeat.items()):
            if (current - last).total_seconds() > self.timeout:
                dead_ids.append(id_)
                del self.last_heartbeat[id_]
        return dead_ids

    def manage(self, connection: Head | Worker) -> set[uuid.UUID]:
        if isinstance(connection, Head):
            self._manage_head(connection)
        elif isinstance(connection, Worker):
            self._manage_worker(connection)
        else:
            raise ValueError("Invalid connection type")

        return set(self._manage_dead())

    def _receive_from_connection(self, id_: uuid.UUID, packet: Optional[CoroutinePacket]) -> Optional[CoroutinePacket]:
        if packet is None:
            return None

        current = datetime.datetime.now(tz=datetime.UTC)
        self.last_heartbeat[id_] = current

        if isinstance(packet.content, ClusterPacket) and isinstance(packet.content.content, HeartbeatContent):
            return None

        return packet

    def _receive_from_head(self, connection: Head) -> dict[uuid.UUID, CoroutinePacket]:
        received_packets = {}
        for id_, packet in connection.receive().items():
            packet = self._receive_from_connection(id_, packet)
            if packet is not None:
                received_packets[id_] = packet
        return received_packets

    def _receive_from_worker(self, connection: Worker) -> dict[uuid.UUID, CoroutinePacket]:
        packet = self._receive_from_connection(connection.id, connection.receive())
        if packet is not None:
            return {connection.id: packet}
        return {}


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
