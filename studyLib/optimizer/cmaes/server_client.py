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


def _proc(gen: int, i: int, ind: base.Individual, env_creator: EnvCreator, queue: mp.Queue, sct: socket.socket):
    received = b""
    request_type = -1
    r_gen = 0
    r_i = 0
    r_score = float("nan")

    try:
        sct.settimeout(600.0)
    except Exception as e:
        print(f"[ERROR(NET1)] {e} (ID:{i})")

    try:
        received = sct.recv(1024)
        while len(received) < 4:
            received = b"".join([received, sct.recv(1)])
        request_type = struct.unpack("<I", received[0:4])[0]
        received = received[4:]
    except Exception as e:
        print(f"[ERROR(NET2)] {e} (ID:{i})")

    if request_type == 1:  # Send Data
        try:
            buf = [struct.pack("<I", gen), struct.pack("<I", i)]
            buf.extend([env_creator.save()])
            buf.extend([struct.pack("<d", x) for x in ind])
            data_bytes = b''.join(buf)
            data_size = len(data_bytes)
            sct.send(struct.pack("<Q", data_size))
            sct.send(data_bytes)
        except Exception as e:
            print(f"[ERROR(NET3)] {e} (ID:{i})")

    elif request_type == 2:  # Receive Result
        size = 4 + 4 + 8
        try:
            while len(received) < size:
                received = b"".join([received, sct.recv(1)])
            r_gen, r_i, r_score = struct.unpack("<IId", received[0:size])
        except Exception as e:
            print(f"[ERROR(NET4)] {e} (ID:{i})")

    else:
        print(f"[ERROR(NET5)] An unknown error occurred. (ID:{i})")

    sct.close()
    queue.put((r_gen, r_i, r_score))


class _ServerProc(base.ProcInterface):
    listener: socket.socket = None

    def __init__(self, gen: int, i: int, ind: base.Individual, env_creator: EnvCreator):
        import select
        import time

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
        self.handle = threading.Thread(target=_proc, args=(gen, i, ind, env_creator, self.queue, sct))
        self.handle.start()
        time.sleep(1)

    def finished(self) -> bool:
        return self.queue.qsize() > 0

    def join(self) -> (int, int, float):
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
    def _treat_sock_error(e):
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
            return ClientCMAES._treat_sock_error(e)
        return ClientCMAES.Result.Succeed, None

    def _request_server(self, sock: socket.socket, request: int):
        try:
            sock.send(struct.pack("<I", request))

        except socket.timeout as e:
            return ClientCMAES.Result.FatalErrorOccurred, e

        except socket.error as e:
            return ClientCMAES._treat_sock_error(e)

        return ClientCMAES.Result.Succeed, None

    def _receive_data(self, sock: socket.socket, default_env_creator: EnvCreator):
        res_type, res_data = self._connect_server(sock)
        if res_type != ClientCMAES.Result.Succeed:
            sock.close()
            return res_type, res_data

        res_type, res_data = self._request_server(sock, 1)
        if res_type != ClientCMAES.Result.Succeed:
            sock.close()
            return res_type, res_data

        try:
            buf = sock.recv(self._buf_size)

            while len(buf) < 8:
                buf = b"".join([buf, sock.recv(1)])
            data_size = struct.unpack("<Q", buf[0:8])[0]
            buf = buf[8:]

            while len(buf) < data_size:
                buf = b"".join([buf, sock.recv(1)])

            gen, i = struct.unpack("<II", buf[0:8])
            buf = buf[8:]

            env_size = default_env_creator.load(buf)
            para = [struct.unpack("<d", buf[i:i + 8])[0] for i in range(env_size, len(buf), 8)]

        except socket.timeout as e:
            return ClientCMAES.Result.Timeout, e

        except socket.error as e:
            return ClientCMAES._treat_sock_error(e)

        return ClientCMAES.Result.Succeed, (gen, i, para, default_env_creator)

    def _return_score(self, sock: socket.socket, score: float, gen: int, i: int):
        res_type, res_data = self._connect_server(sock)
        if res_type != ClientCMAES.Result.Succeed:
            sock.close()
            return res_type, res_data

        res_type, res_data = self._request_server(sock, 2)
        if res_type != ClientCMAES.Result.Succeed:
            sock.close()
            return res_type, res_data

        try:
            sock.send(struct.pack("<I", gen))
            sock.send(struct.pack("<I", i))
            sock.send(struct.pack("<d", score))

        except socket.timeout as e:
            return ClientCMAES.Result.FatalErrorOccurred, e

        except socket.error as e:
            return ClientCMAES._treat_sock_error(e)

        return ClientCMAES.Result.Succeed, None

    def optimize(self, default_env_creator: EnvCreator, timeout: float = 60.0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        res_type, res_data = self._receive_data(sock, default_env_creator)
        if res_type != ClientCMAES.Result.Succeed:
            sock.close()
            return res_type, res_data
        sock.close()

        gen, i, para, env_creator = res_data
        env = env_creator.create(para)
        score = env.calc()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        res_type, e = self._return_score(sock, gen, i, score)
        if res_type != ClientCMAES.Result.Succeed:
            sock.close()
            return res_type, e
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
        res_type, res_data = self._receive_data(sock, default_env_creator)
        if res_type != ClientCMAES.Result.Succeed:
            sock.close()
            return res_type, res_data
        sock.close()

        gen, i, para, env_creator = res_data
        env = env_creator.create_mujoco_env(para, window, camera)
        score = env.calc_and_show()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        res_type, e = self._return_score(sock, gen, i, score)
        if res_type != ClientCMAES.Result.Succeed:
            sock.close()
            return res_type, e
        sock.close()

        return ClientCMAES.Result.Succeed, (para, env)
