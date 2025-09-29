import datetime
import uuid
from typing import Optional

from icecream import ic

from ...prelude import *
from ._head import Head
from ._worker import Worker


class ConnectionManager:
    def __init__(self, interval: float = 10.0, timeout: float = 30.0):
        self.last_receive_heartbeat: dict[uuid.UUID, datetime.datetime] = {}
        self.last_send_heartbeat: dict[uuid.UUID, datetime.datetime] = {}
        self.interval = interval
        self.timeout = timeout

    def _manage_heartbeat(self, id_: uuid.UUID) -> bool:
        current = datetime.datetime.now(tz=datetime.UTC)
        if id_ not in self.last_send_heartbeat:
            self.last_send_heartbeat[id_] = current
            return True
        do_send_heartbeat = (current - self.last_send_heartbeat[id_]).total_seconds() > self.interval
        if do_send_heartbeat:
            self.last_send_heartbeat[id_] = current
        return do_send_heartbeat

    async def _manage_head(self, connection: Head):
        for i in connection.get_ids():
            if self._manage_heartbeat(i):
                heartbeat = HeartbeatContent()
                packet = ic(ClusterPacket(ClusterPacketType.HEARTBEAT, heartbeat))
                await connection.send(i, packet)

    async def _manage_worker(self, connection: Worker):
        id_ = connection.id
        if self._manage_heartbeat(id_):
            heartbeat = HeartbeatContent()
            packet = ic(ClusterPacket(ClusterPacketType.HEARTBEAT, heartbeat))
            await connection.send(packet)

    def _manage_dead(self) -> list[uuid.UUID]:
        dead_ids = []
        current = datetime.datetime.now(tz=datetime.UTC)
        for id_, last in list(self.last_receive_heartbeat.items()):
            if ic((current - last).total_seconds() > self.timeout):
                dead_ids.append(ic(id_))
                del self.last_receive_heartbeat[id_]
        return dead_ids

    async def manage(self, connection: Head | Worker):
        if isinstance(connection, Head):
            await self._manage_head(connection)
        elif isinstance(connection, Worker):
            await self._manage_worker(connection)
        else:
            raise ValueError("Invalid connection type")

        self._manage_dead()

    def _receive_from_head(self, connection: Head) -> dict[uuid.UUID, list[ClusterPacket]]:
        current = datetime.datetime.now(tz=datetime.UTC)
        received_packets = {}

        for id_, cluster_packet_list in connection.receive().items():
            received_packets[id_] = []
            self.last_receive_heartbeat[id_] = current

            for cluster_packet in cluster_packet_list:
                if isinstance(cluster_packet.content, HeartbeatContent):
                    continue
                received_packets[id_].append(cluster_packet)

        return received_packets

    def _receive_from_worker(self, connection: Worker) -> dict[uuid.UUID, list[ClusterPacket]]:
        current = datetime.datetime.now(tz=datetime.UTC)
        packet = connection.receive()

        if packet is None:
            return {}

        self.last_receive_heartbeat[connection.id] = current

        if isinstance(packet.content, HeartbeatContent):
            return {}

        return {connection.id: [packet]}

    def receive(self, connection: Head | Worker) -> dict[uuid.UUID, list[ClusterPacket]]:
        if isinstance(connection, Head):
            return self._receive_from_head(connection)
        elif isinstance(connection, Worker):
            return self._receive_from_worker(connection)
        else:
            raise ValueError("Invalid connection type")

    def get_ids(self) -> set[uuid.UUID]:
        return set(self.last_receive_heartbeat.keys())
