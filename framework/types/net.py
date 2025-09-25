import enum


class ClusterPacket:
    def __init__(self, type_: str, content):
        self.type: str = type_
        self.content = content

    def __repr__(self):
        return f"<ClusterPacket type={self.type} content={self.content}>"


class CoroutinePacketType(enum.Enum):
    STOP = 1
    CLUSTER_PACKET = 2


class CoroutinePacket:
    def __init__(self, type_: CoroutinePacketType, content):
        self.type: CoroutinePacketType = type_
        self.content = content

    def __repr__(self):
        return f"<AsyncTaskPacket type={self.type.name} content={self.content}>"

    @classmethod
    def stop_packet(cls):
        return cls(CoroutinePacketType.STOP, None)

    @classmethod
    def worker_packet(cls, content: ClusterPacket):
        return cls(CoroutinePacketType.CLUSTER_PACKET, content)
