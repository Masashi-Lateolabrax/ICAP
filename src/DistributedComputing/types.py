from enum import Enum


class NetworkResultType(Enum):
    Succeed = 0
    Other = 1
    Timeout = 2
    FailedToGetBufSize = 3


class NetworkResult:
    def __init__(self, type_: NetworkResultType, cxt=None):
        self.type = type_
        self.cxt = cxt
