import socket
import pickle
import struct
import logging
from typing import Optional

from icecream import ic
from ..prelude import *


def _send_message(sock: socket.socket, data: bytes, attempt_count: int = 10) -> CommunicationResult:
    size = struct.pack('!I', len(data))

    for attempt in range(attempt_count):
        try:
            sock.sendall(size + data)
            return CommunicationResult.SUCCESS

        except socket.timeout:
            logging.warning(f"Socket timeout on attempt {attempt + 1} while sending message")
            continue

        except (socket.error, struct.error, BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            logging.error(f"Error sending message: {e}")
            return CommunicationResult.CONNECTION_ERROR

    logging.error("Failed to send message after multiple attempts")
    return CommunicationResult.OVER_ATTEMPT_COUNT


def _receive_bytes(sock: socket.socket, size: int, retry: int = 0) -> tuple[CommunicationResult, Optional[bytes]]:
    attempt = 0
    data = b''
    while len(data) < size:
        try:
            chunk = sock.recv(size - len(data))
            if not chunk:
                logging.info("Server disconnected")
                return CommunicationResult.DISCONNECTED, None
            data += chunk

        except socket.timeout:
            if retry > 0:
                if attempt >= retry:
                    logging.error("Exceeded maximum retry attempts while receiving data")
                    return CommunicationResult.OVER_ATTEMPT_COUNT, None

                attempt += 1
                logging.warning(f"Socket timeout on attempt {attempt} while receiving data")

                continue

            logging.error("Socket timeout while receiving data")
            return CommunicationResult.CONNECTION_ERROR, None

        except (socket.error, struct.error, ConnectionResetError, ConnectionAbortedError) as e:
            logging.error(f"Connection error while receiving data: {e}")
            return CommunicationResult.CONNECTION_ERROR, None

    return CommunicationResult.SUCCESS, data


def send_packet(sock: socket.socket, packet: Packet, retry: int = 10) -> CommunicationResult:
    try:
        data = pickle.dumps(packet)
        return _send_message(sock, data, attempt_count=retry)
    except pickle.PicklingError as e:
        logging.error(f"Pickle error while sending packet: {e}")
        return CommunicationResult.BROKEN_DATA


def receive_packet(sock: socket.socket, retry: int = 0) -> tuple[CommunicationResult, Optional[Packet]]:
    result, raw = _receive_bytes(sock, 4, retry)
    if result != CommunicationResult.SUCCESS:
        return result, None

    try:
        size_data = struct.unpack('!I', raw)[0]
    except struct.error as e:
        logging.error(f"Error unpacking size data: {e}")
        return CommunicationResult.BROKEN_DATA, None

    result, raw = _receive_bytes(sock, size_data, retry)
    if result != CommunicationResult.SUCCESS:
        return result, None

    try:
        packet = pickle.loads(raw)
        return CommunicationResult.SUCCESS, packet
    except pickle.UnpicklingError as e:
        logging.error(f"Pickle error while receiving packet: {e}")
        return CommunicationResult.BROKEN_DATA, None


def communicate(
        sock: socket.socket, packet: Packet, retry_count: int = 10
) -> tuple[CommunicationResult, Optional[Packet]]:
    try:
        result = send_packet(sock, packet, retry_count)
        if result != CommunicationResult.SUCCESS:
            logging.error("Failed to send packet")
            return result, None
        ic("Packet sent successfully")

    except socket.error as e:
        logging.error(f"Socket error during communication: {e}")
        return CommunicationResult.CONNECTION_ERROR, None

    try:
        result, ack_packet = receive_packet(sock, retry_count)
        if result != CommunicationResult.SUCCESS:
            logging.error("Failed to receive ACK packet")
            return result, None

        if ack_packet is None or ack_packet.packet_type != PacketType.ACK:
            logging.error("Received invalid ACK packet")
            return CommunicationResult.CONNECTION_ERROR, None
        ic("ACK packet received successfully")

    except socket.error as e:
        logging.error(f"Socket error during heartbeat ACK: {e}")
        return CommunicationResult.CONNECTION_ERROR, None

    return CommunicationResult.SUCCESS, ack_packet
