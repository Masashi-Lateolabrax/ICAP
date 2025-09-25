import asyncio

from icecream import ic

from ...prelude import *
from ._tunnel import Tunnel


async def send_payload(writer: asyncio.StreamWriter, payload: ClusterPacket):
    payload_bytes = payload.as_bytes()
    payload_size = len(payload_bytes)
    if payload_size < 4:
        raise ValueError("Payload size must be at least 4 bytes.")

    writer.write(payload_size.to_bytes(4, byteorder='big'))
    writer.write(payload_bytes)
    ic(await writer.drain())


async def receive_payload(reader: asyncio.StreamReader, timeout: float) -> ClusterPacket:
    size_data = ic(await asyncio.wait_for(reader.readexactly(4), timeout=timeout))
    payload_size = int.from_bytes(size_data, byteorder='big')
    if payload_size < 4:
        raise ValueError("Payload size must be at least 4 bytes.")

    payload_data = ic(await asyncio.wait_for(reader.readexactly(payload_size), timeout=timeout))
    return ClusterPacket.from_bytes(payload_data)


async def _relay_local_to_remote(writer: asyncio.StreamWriter, tunnel: Tunnel):
    packet: CoroutinePacket = ic(tunnel.receive())
    if packet is None:
        return False

    if packet.type == CoroutinePacketType.STOP:
        return True

    if packet.type == CoroutinePacketType.CLUSTER_PACKET:
        payload = ic(packet.content)
        if not isinstance(payload, ClusterPacket):
            raise ValueError("Invalid packet content type. Expected WorkerPacket for PAYLOAD type.")
        await send_payload(writer, payload)

    return False


async def _relay_remote_to_local(reader: asyncio.StreamReader, tunnel: Tunnel, timeout: float):
    try:
        response: CoroutinePacket = ic(await receive_payload(reader, timeout))
    except asyncio.TimeoutError:
        return

    response_packet = ic(CoroutinePacket(CoroutinePacketType.CLUSTER_PACKET, response))
    await tunnel.send(response_packet)


async def relay_routine(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter, tunnel: Tunnel, timeout: float
) -> bool:
    stop_signal = ic(await _relay_local_to_remote(writer, tunnel))
    await _relay_remote_to_local(reader, tunnel, timeout)
    return not stop_signal
