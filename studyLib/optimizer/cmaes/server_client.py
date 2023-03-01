import array
import enum
import multiprocessing as mp
import platform
import socket
import struct
import threading

from studyLib.miscellaneous import Window
from studyLib.wrap_mjc import Camera
from studyLib.optimizer import Hist, EnvCreator, MuJoCoEnvCreator
from studyLib.optimizer.cmaes import base


def _proc(i: int, ind: base.Individual, env_creator: EnvCreator, queue: mp.Queue, sct: socket.socket):
    buf = [env_creator.save()]
    buf.extend([struct.pack("<d", x) for x in ind])
    data_bytes = b''.join(buf)
    data_size = len(data_bytes)
    try:
        sct.settimeout(600.0)
        sct.send(struct.pack("<Q", data_size))
        sct.send(data_bytes)
        received = sct.recv(1024)
        score = struct.unpack("<d", received)[0]
    except Exception as e:
        print(f"[ERROR] {e} (ID:{i})")
        score = float("nan")
    sct.close()
    queue.put(score)


class _ServerProc(base.ProcInterface):
    listener: socket.socket = None

    def __init__(self, i: int, ind: base.Individual, env_creator: EnvCreator):
        import select

        r = []
        for ec in range(10):
            r, _, _ = select.select([self.listener], [], [], 30)
            if len(r) != 0:
                break
            print(f"[WARNING({ec + 1}/10)] Clients don't be coming.")
        if len(r) == 0:
            raise socket.timeout

        sct, addr = self.listener.accept()

        self.queue = mp.Queue(1)
        self.handle = threading.Thread(target=_proc, args=(i, ind, env_creator, self.queue, sct))
        self.handle.start()

    def finished(self) -> bool:
        return self.queue.qsize() > 0

    def join(self) -> float:
        self.handle.join()
        return self.queue.get()


class ServerCMAES:
    def __init__(
            self,
            port: int,
            dim: int,
            generation: int,
            population: int,
            mu: int = -1,
            sigma: float = 0.3,
            centroid=None,
            cmatrix=None,
            minimalize: bool = True,
    ):
        self._base = base.BaseCMAES(dim, population, mu, sigma, centroid, minimalize, population, cmatrix)
        self._generation = generation

        _ServerProc.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _ServerProc.listener.bind(("", port))
        _ServerProc.listener.listen(10)

    def get_best_para(self) -> array.array:
        return self._base.get_best_para()

    def get_best_score(self) -> float:
        return self._base.get_best_score()

    def get_history(self) -> Hist:
        return self._base.get_history()

    def set_start_handler(self, handler=base.default_start_handler):
        self._base.set_start_handler(handler)

    def set_end_handler(self, handler=base.default_end_handler):
        self._base.set_end_handler(handler)

    def optimize(self, env_creator: EnvCreator):
        for gen in range(1, self._generation + 1):
            self._base.optimize_current_generation(gen, self._generation, env_creator, _ServerProc)


class ClientCMAES:
    class Result(enum.Enum):
        Succeed = 1
        ErrorOccurred = 2
        FatalErrorOccurred = 3
        Timeout = 4

    def __init__(self, address, port, buf_size: int = 1024):
        self._address = address
        self._port = port
        self._buf_size = buf_size

    @staticmethod
    def _treat_sock_error(sock: socket.socket, e):
        os = platform.system()
        if os == "Windows":
            if e.errno == 10054:  # [WinError 10054] 既存の接続はリモート ホストに強制的に切断されました。
                return ClientCMAES.Result.FatalErrorOccurred, e
            elif e.errno == 10057:  # [WinError 10057] ソケットが接続されていないか、sendto呼び出しを使ってデータグラムソケットで...
                return ClientCMAES.Result.FatalErrorOccurred, e
            elif e.errno == 10060:  # [WinError 10060] 接続済みの呼び出し先が一定時間を過ぎても正しく応答しなかったため...
                return ClientCMAES.Result.Timeout, e
            elif e.errno == 10061:  # [WinError 10061] 対象のコンピューターによって拒否されたため、接続できませんでした。
                return ClientCMAES.Result.ErrorOccurred, e
        return ClientCMAES.Result.FatalErrorOccurred, e

    def _connect_server(self, sock: socket.socket):
        try:
            sock.connect((self._address, self._port))
        except socket.timeout as e:
            return ClientCMAES.Result.Timeout, e
        except socket.error as e:
            return ClientCMAES._treat_sock_error(sock, e)
        return ClientCMAES.Result.Succeed, None

    def _receive_data(self, sock: socket.socket, default_env_creator: EnvCreator):
        try:
            buf = b""
            while len(buf) < 8:
                buf = b"".join([buf, sock.recv(8)])
            data_size = struct.unpack("<Q", buf[0:8])[0]

            buf = buf[8:]
            while len(buf) < data_size:
                buf = b"".join([buf, sock.recv(self._buf_size)])

            env_size = default_env_creator.load(buf)
            para = [struct.unpack("<d", buf[i:i + 8])[0] for i in range(env_size, len(buf), 8)]

        except socket.timeout as e:
            return ClientCMAES.Result.Timeout, e

        except socket.error as e:
            return ClientCMAES._treat_sock_error(sock, e)

        return ClientCMAES.Result.Succeed, (para, default_env_creator)

    def _return_score(self, sock: socket.socket, score: float):
        try:
            sock.send(struct.pack("<d", score))

        except socket.timeout as e:
            return ClientCMAES.Result.FatalErrorOccurred, e

        except socket.error as e:
            return ClientCMAES._treat_sock_error(sock, e)

        return ClientCMAES.Result.Succeed, None

    def optimize(self, default_env_creator: EnvCreator, timeout: float = 60.0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        res, pe = self._connect_server(sock)
        if res != ClientCMAES.Result.Succeed:
            sock.close()
            return res, pe

        res, pe = self._receive_data(sock, default_env_creator)

        if res != ClientCMAES.Result.Succeed:
            sock.close()
            return res, pe

        para = pe[0]
        env_creator: EnvCreator = pe[1]
        env = env_creator.create(para)
        score = env.calc()

        res, e = self._return_score(sock, score)

        if res != ClientCMAES.Result.Succeed:
            sock.close()
            return res, e

        sock.close()
        return ClientCMAES.Result.Succeed, (para, env)

    def optimize_and_show(
            self,
            default_env_creator: MuJoCoEnvCreator,
            window: Window, camera: Camera,
            timeout: float = 60.0
    ):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        res, pe = self._connect_server(sock)
        if res != ClientCMAES.Result.Succeed:
            sock.close()
            return res, pe

        res, pe = self._receive_data(sock, default_env_creator)

        if res != ClientCMAES.Result.Succeed:
            sock.close()
            return res, pe

        para = pe[0]
        env_creator: MuJoCoEnvCreator = pe[1]
        env = env_creator.create_mujoco_env(para, window, camera)
        score = env.calc_and_show()

        res, e = self._return_score(sock, score)

        if res != ClientCMAES.Result.Succeed:
            sock.close()
            return res, e

        sock.close()
        return ClientCMAES.Result.Succeed, (para, env)
