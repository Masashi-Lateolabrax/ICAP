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

    async def manage(self, connection: Head | Worker) -> set[uuid.UUID]:
        if isinstance(connection, Head):
            await self._manage_head(connection)
        elif isinstance(connection, Worker):
            await self._manage_worker(connection)
        else:
            raise ValueError("Invalid connection type")

        return set(self._manage_dead())

    def _receive_from_connection(self, id_: uuid.UUID, packet: Optional[ClusterPacket]) -> Optional[ClusterPacket]:
        if packet is None:
            return None

        current = datetime.datetime.now(tz=datetime.UTC)
        self.last_receive_heartbeat[id_] = current

        if isinstance(packet.content, ClusterPacket) and isinstance(packet.content.content, HeartbeatContent):
            return None

        return packet

    def _receive_from_head(self, connection: Head) -> dict[uuid.UUID, ClusterPacket]:
        received_packets = {}
        for id_, packet in connection.receive().items():
            packet = self._receive_from_connection(id_, packet)
            if packet is not None:
                received_packets[id_] = packet
        return received_packets

    def _receive_from_worker(self, connection: Worker) -> dict[uuid.UUID, ClusterPacket]:
        packet = self._receive_from_connection(connection.id, connection.receive())
        if packet is not None:
            return {connection.id: packet}
        return {}

    def receive(self, connection: Head | Worker) -> dict[uuid.UUID, ClusterPacket]:
        if isinstance(connection, Head):
            return self._receive_from_head(connection)
        elif isinstance(connection, Worker):
            return self._receive_from_worker(connection)
        else:
            raise ValueError("Invalid connection type")

    def get_ids(self) -> set[uuid.UUID]:
        return set(self.last_receive_heartbeat.keys())
