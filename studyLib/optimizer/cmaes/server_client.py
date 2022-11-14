import array
import copy
import datetime
import multiprocessing as mp
import platform
import socket
from studyLib.miscellaneous import Window, Recorder
from studyLib.wrap_mjc import Camera
from studyLib.optimizer.cmaes.base import Proc, Individual, BaseCMAES, default_end_handler, default_start_handler
from studyLib.optimizer import EnvInterface, Hist, MuJoCoEnvInterface


class ServerProc(Proc):
    def __init__(self, env: EnvInterface, listener: socket.socket):
        super().__init__(env)
        self.env = env
        self.listener = listener
        self.soc: socket.socket = None

    def ready(self):
        soc, _addr = self.listener.accept()
        self.soc = soc

    def start(self, index: int, queue: mp.Queue, ind: Individual):
        import struct
        buf = [self.env.save()]
        buf.extend([struct.pack("<d", x) for x in ind])
        try:
            self.soc.send(b''.join(buf))
            received = self.soc.recv(1024)
            score = struct.unpack("<d", received)[0]
        except Exception as e:
            print(e)
            score = float("nan")
        queue.put((index, score))


class ServerCMAES:
    def __init__(
            self,
            port: int,
            dim: int,
            generation: int,
            population: int,
            mu: int = -1,
            sigma: float = 0.3,
            minimalize: bool = True
    ):
        self._base = BaseCMAES(dim, population, mu, sigma, minimalize, population)
        self._generation = generation
        self._port = port

    def get_best_para(self) -> array.array:
        return self._base.get_best_para()

    def get_best_score(self) -> float:
        return self.get_best_score()

    def get_history(self) -> Hist:
        return self._base.get_history()

    def set_start_handler(self, handler=default_start_handler):
        self._base.set_start_handler(handler)

    def set_end_handler(self, handler=default_end_handler):
        self._base.set_end_handler(handler)

    def optimize(self, env: EnvInterface):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("", self._port))
        listener.listen(2)

        for gen in range(1, self._generation + 1):
            proc = ServerProc(copy.deepcopy(env), listener)
            self._base.optimize_current_generation(gen, self._generation, proc)

    def optimize_with_recoding_min(self, env: MuJoCoEnvInterface, window: Window, camera: Camera):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("", self._port))
        listener.listen(2)

        for gen in range(1, self._generation + 1):
            proc = ServerProc(copy.deepcopy(env), listener)
            good_para = self._base.optimize_current_generation(gen, self._generation, proc)

            time = datetime.datetime.now()
            recorder = Recorder(f"{gen}({time.strftime('%y%m%d_%H%M%S')}).mp4", 30, 640, 480)
            window.set_recorder(recorder)
            env.calc_and_show(good_para, window, camera)
            window.set_recorder(None)


class ClientCMAES:
    def __init__(self, address, port, buf_size: int = 1024):
        self._address = address
        self._port = port
        self._buf_size = buf_size

    def optimize(self, env: EnvInterface):
        import socket
        import struct

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self._address, self._port))

            received = sock.recv(self._buf_size)
            print(f"receive data size : {len(received)}/{self._buf_size}")

            env_size = env.load(received)
            para = [struct.unpack("<d", received[i:i + 8])[0] for i in range(env_size, len(received), 8)]

            score = env.calc(para)

            sock.send(struct.pack("<d", score))
            print(f"score : {score}")

            sock.shutdown(socket.SHUT_RDWR)
            sock.close()

        except socket.error as e:
            os = platform.system()
            if os == "Windows":
                if e.errno == 10054:  # [WinError 10054] 既存の接続はリモート ホストに強制的に切断されました。
                    sock.close()
                    return e, False
                elif e.errno == 10057:  # [WinError 10057] ソケットが接続されていないか、sendto呼び出しを使ってデータグラムソケットで...
                    sock.close()
                    return e, False
                elif e.errno == 10060:  # [WinError 10060] 接続済みの呼び出し先が一定時間を過ぎても正しく応答しなかったため...
                    sock.close()
                    return e, True
                elif e.errno == 10061:  # [WinError 10061] 対象のコンピューターによって拒否されたため、接続できませんでした。
                    sock.close()
                    return e, True

            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            return e, False

        return None, True
