from . import *


def launch_client_process(address, port, processor) -> (int, bytes):
    nm = NetManager(address, port, 10)

    for i in range(0, 5):
        res = nm.receive()

        match res.type:
            case NetworkResultType.Succeed:
                return res.cxt
            case NetworkResultType.Timeout:
                continue
            case _:
                raise "Failed to receive data."
