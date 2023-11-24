from . import *


class NetManager:
    BUFFER_SIZE = 3702

    def __init__(self, address, port, timeout):
        self._address = address
        self._port = port
        self._timeout = timeout

    def _create_connection(self) -> NetworkResult:
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._timeout)
            sock.connect((self._address, self._port))

        except socket.timeout as e:
            return NetworkResult(NetworkResultType.Timeout, e)
        except Exception as e:
            return NetworkResult(NetworkResultType.Other, e)

        return NetworkResult(NetworkResultType.Succeed, sock)

    def receive(self) -> NetworkResult:
        import socket
        import struct

        # Create connection.
        sock: NetworkResult = self._create_connection()
        if sock.type != NetworkResultType.Succeed:
            return sock
        sock: socket.socket = sock.cxt

        # Get data size.
        try:
            buf = sock.recv(NetManager.BUFFER_SIZE)
            res = struct.unpack("<II", buf[0:8])
            prefix = res[0]
            buf_size = res[1]

        except socket.timeout as e:
            return NetworkResult(NetworkResultType.Timeout, e)
        except Exception as e:
            return NetworkResult(NetworkResultType.Other, e)
        if buf_size is None:
            return NetworkResult(NetworkResultType.FailedToGetBufSize)

        # Get data.
        buf = bytearray([0] * buf_size)
        try:
            i = 0
            while i < buf_size - NetManager.BUFFER_SIZE:
                b = sock.recv(NetManager.BUFFER_SIZE)
                buf[i:i + len(b)] = b
                i += len(b)
            while i < buf_size:
                b = sock.recv(buf_size - i)
                buf[i:i + len(b)] = b
                i += len(b)
            sock.close()

        except socket.timeout as e:
            return NetworkResult(NetworkResultType.Timeout, e)
        except Exception as e:
            return NetworkResult(NetworkResultType.Other, e)

        return NetworkResult(NetworkResultType.Succeed, (prefix, buf))

    def send(self, prefix: int, cxt: bytes) -> NetworkResult:
        import socket
        import struct

        # Create connection.
        sock: NetworkResult = self._create_connection()
        if sock.type != NetworkResultType.Succeed:
            return sock
        sock: socket.socket = sock.cxt

        # Send data.
        try:
            sock.send(
                struct.pack("<II", int(prefix), len(cxt))
            )
            sock.send(cxt)
            sock.close()

        except socket.timeout as e:
            return NetworkResult(NetworkResultType.Timeout, e)
        except Exception as e:
            return NetworkResult(NetworkResultType.Other, e)

        return NetworkResult(NetworkResultType.Succeed)
