import array
import datetime
import multiprocessing as mp
import platform
import socket
import threading

from studyLib.miscellaneous import Window, Recorder
from studyLib.wrap_mjc import Camera
from studyLib.optimizer.cmaes.base import Individual, BaseCMAES, ProcInterface, default_end_handler, \
    default_start_handler
from studyLib.optimizer import Hist, EnvCreator, MuJoCoEnvCreator


def _proc(ind: Individual, env_creator: EnvCreator, queue: mp.Queue, sct: socket.socket):
    import struct
    buf = [env_creator.save()]
    buf.extend([struct.pack("<d", x) for x in ind])
    try:
        sct.send(b''.join(buf))
        received = sct.recv(1024)
        score = struct.unpack("<d", received)[0]
    except Exception as e:
        print(e)
        score = float("nan")
    queue.put(score)


class _ServerProc(ProcInterface):
    listener: socket.socket = None

    def __init__(self, ind: Individual, env_creator: EnvCreator):
        self.queue = mp.Queue(1)
        sct, _addr = self.listener.accept()
        self.handle = threading.Thread(target=_proc, args=(ind, env_creator, self.queue, sct))
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
            minimalize: bool = True
    ):
        self._base = BaseCMAES(dim, population, mu, sigma, minimalize, population)
        self._generation = generation

        _ServerProc.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _ServerProc.listener.bind(("", port))
        _ServerProc.listener.listen(2)

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

    def optimize(self, env_creator: EnvCreator):
        for gen in range(1, self._generation + 1):
            self._base.optimize_current_generation(gen, self._generation, env_creator, _ServerProc)

    def optimize_with_recoding_min(self, env_creator: MuJoCoEnvCreator, window: Window, camera: Camera):
        for gen in range(1, self._generation + 1):
            good_para = self._base.optimize_current_generation(gen, self._generation, env_creator, _ServerProc)

            time = datetime.datetime.now()
            recorder = Recorder(f"{gen}({time.strftime('%y%m%d_%H%M%S')}).mp4", 30, 640, 480)
            window.set_recorder(recorder)
            env = env_creator.create_mujoco_env()
            env.calc_and_show(good_para, window, camera)
            window.set_recorder(None)


class ClientCMAES:
    def __init__(self, address, port, buf_size: int = 1024):
        self._address = address
        self._port = port
        self._buf_size = buf_size

    def optimize(self, default_env_creator: EnvCreator):
        import socket
        import struct

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self._address, self._port))

            received = sock.recv(self._buf_size)
            print(f"receive data size : {len(received)}/{self._buf_size}")

            env_size = default_env_creator.load(received)
            para = [struct.unpack("<d", received[i:i + 8])[0] for i in range(env_size, len(received), 8)]

            env = default_env_creator.create()
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