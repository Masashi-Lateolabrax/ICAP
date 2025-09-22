import asyncio

from .packets import WorkerPacket


async def send_payload(writer: asyncio.StreamWriter, payload: WorkerPacket):
    payload_bytes = payload.as_bytes()
    payload_size = len(payload_bytes)
    if payload_size < 4:
        raise ValueError("Payload size must be at least 4 bytes.")

    writer.write(payload_size.to_bytes(4, byteorder='big'))
    writer.write(payload_bytes)
    await writer.drain()


async def receive_payload(reader: asyncio.StreamReader, timeout: float) -> WorkerPacket:
    size_data = await asyncio.wait_for(reader.readexactly(4), timeout=timeout)
    payload_size = int.from_bytes(size_data, byteorder='big')
    if payload_size < 4:
        raise ValueError("Payload size must be at least 4 bytes.")

    payload_data = await asyncio.wait_for(reader.readexactly(payload_size), timeout=timeout)
    return WorkerPacket.from_bytes(payload_data)
