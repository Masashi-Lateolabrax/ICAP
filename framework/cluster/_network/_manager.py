import datetime
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

    def receive(self, connection: Head | Worker) -> dict[uuid.UUID, CoroutinePacket]:
        if isinstance(connection, Head):
            return self._receive_from_head(connection)
        elif isinstance(connection, Worker):
            return self._receive_from_worker(connection)
        else:
            raise ValueError("Invalid connection type")

    def get_ids(self) -> set[uuid.UUID]:
        return set(self.last_heartbeat.keys())
