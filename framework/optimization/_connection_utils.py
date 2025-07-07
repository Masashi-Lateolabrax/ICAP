import socket
import pickle
import struct
import logging
from typing import Optional

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


def send_individuals(sock: socket.socket, individuals: list[Individual]) -> bool:
    """Send a list of Individual objects over socket."""
    try:
        data = pickle.dumps(individuals)
        result = _send_message(sock, data)
        if result != _CommunicationResult.SUCCESS:
            logging.error("Connection error while sending individual list")
        return result == _CommunicationResult.SUCCESS

    except pickle.PicklingError as e:
        logging.error(f"Pickle error while sending individual list: {e}")
        return False


def receive_individuals(sock: socket.socket, blocking=True) -> tuple[bool, Optional[list[Individual]]]:
    """Receive a list of Individual objects from socket with proper error handling.
    
    Returns:
        tuple[bool, Optional[list[Individual]]]: Status and received objects:
            - (True, list[Individual]): Successfully received and deserialized Individual objects
            - (True, None): Operation timed out (exceeded retry attempts) - recoverable
            - (False, None): Fatal error occurred (connection, protocol, or deserialization error)
    """
    # First, receive the size
    result, raw = _receive_bytes(sock, 4, blocking=blocking)
    if result == _CommunicationResult.OVER_ATTEMPT_COUNT:
        return True, None
    elif result != _CommunicationResult.SUCCESS:
        return False, None

    # Extract size from the 4-byte data
    try:
        size_data = struct.unpack('!I', raw)[0]
    except struct.error as e:
        logging.error(f"Error unpacking size data: {e}")
        return False, None

    # Receive the actual data
    result, raw = _receive_bytes(sock, size_data, blocking=blocking)
    if result == _CommunicationResult.OVER_ATTEMPT_COUNT:
        return True, None
    elif result != _CommunicationResult.SUCCESS:
        return False, None

    # Deserialize the Individual list
    try:
        individuals = pickle.loads(raw)
        return True, individuals
    except pickle.UnpicklingError as e:
        logging.error(f"Pickle error while receiving individual list: {e}")
        return False, None


def send_packet(sock: socket.socket, packet: Packet, attempt_count: int = 10) -> CommunicationResult:
    try:
        data = pickle.dumps(packet)
        return _send_message(sock, data, attempt_count=attempt_count)
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
