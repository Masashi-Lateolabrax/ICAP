import socket
import pickle
import struct
import logging
from typing import Optional
import enum

from ._types import ATTEMPT_COUNT
from ._types import Individual


class _CommunicationResult(enum.Enum):
    SUCCESS = 0
    OVER_ATTEMPT_COUNT = 1
    CONNECTION_ERROR = 2
    DISCONNECTED = 3
    BROKEN_DATA = 4


def _send_message(sock: socket.socket, data: bytes) -> _CommunicationResult:
    """Send a message with size prefix over socket."""
    size = struct.pack('!I', len(data))

    for attempt in range(ATTEMPT_COUNT):
        try:
            sock.sendall(size + data)
            return _CommunicationResult.SUCCESS

        except socket.timeout:
            logging.warning(f"Socket timeout on attempt {attempt + 1} while sending message")
            continue

        except (socket.error, struct.error, BrokenPipeError) as e:
            logging.error(f"Error sending message: {e}")
            return _CommunicationResult.CONNECTION_ERROR

    logging.error("Failed to send message after multiple attempts")
    return _CommunicationResult.OVER_ATTEMPT_COUNT


def _receive_bytes(sock: socket.socket, size: int) -> tuple[_CommunicationResult, Optional[bytes]]:
    attempt = 0
    data = b''
    while len(data) < size:
        try:
            chunk = sock.recv(size - len(data))
            if not chunk:
                logging.info("Server disconnected")
                return _CommunicationResult.DISCONNECTED, None
            data += chunk

        except socket.timeout:
            attempt += 1
            logging.warning(f"Socket timeout on attempt {attempt} while receiving data")
            if attempt >= ATTEMPT_COUNT:
                logging.error("Exceeded maximum attempt count while receiving data")
                return _CommunicationResult.OVER_ATTEMPT_COUNT, None
            continue

        except (socket.error, struct.error) as e:
            logging.error(f"Connection error while receiving data: {e}")
            return _CommunicationResult.CONNECTION_ERROR, None

    return _CommunicationResult.SUCCESS, data


def send_individual(sock: socket.socket, individual: Individual) -> bool:
    """Send an Individual object over socket."""
    try:
        data = pickle.dumps(individual)
        result = _send_message(sock, data)
        if result != _CommunicationResult.SUCCESS:
            logging.error("Connection error while sending individual")
        return result == _CommunicationResult.SUCCESS

    except pickle.PicklingError as e:
        logging.error(f"Pickle error while sending individual: {e}")
        return False


def receive_individual(sock: socket.socket) -> tuple[bool, Optional[Individual]]:
    """Receive an Individual object from socket with proper error handling.
    
    This function performs a two-phase receive: first the message size (4 bytes),
    then the pickled Individual object data. It handles timeouts gracefully and
    provides detailed error logging for debugging.
    
    Returns:
        tuple[bool, Optional[Individual]]: Status and received object:
            - (True, Individual): Successfully received and deserialized Individual object
            - (True, None): Operation timed out (exceeded retry attempts) - recoverable
            - (False, None): Fatal error occurred (connection, protocol, or deserialization error)
    """
    # First, receive the size
    result, raw = _receive_bytes(sock, 4)
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
    result, raw = _receive_bytes(sock, size_data)
    if result == _CommunicationResult.OVER_ATTEMPT_COUNT:
        return True, None
    elif result != _CommunicationResult.SUCCESS:
        return False, None

    # Deserialize the Individual object
    try:
        individual = pickle.loads(raw)
        return True, individual
    except pickle.UnpicklingError as e:
        logging.error(f"Pickle error while receiving individual: {e}")
        return False, None
