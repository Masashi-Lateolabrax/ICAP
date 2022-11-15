import array
import datetime
import multiprocessing as mp
from studyLib.wrap_mjc import Camera
from studyLib.miscellaneous import Window, Recorder
from studyLib.optimizer import Hist, EnvCreator, MuJoCoEnvCreator
from studyLib.optimizer.cmaes.base import Individual, BaseCMAES, ProcInterface, default_start_handler, \
    default_end_handler


def _func(ind: Individual, env_creator: EnvCreator, queue: mp.Queue):
    env = env_creator.create()
    score = env.calc(ind)
    queue.put(score)


class _ThreadProc(ProcInterface):
    def __init__(self, ind: Individual, env_creator: EnvCreator):
        self.queue = mp.Queue(1)
        self.handle = mp.Process(target=_func, args=(ind, env_creator, self.queue))
        self.handle.start()

    def finished(self) -> bool:
        return self.queue.qsize() > 0

    def join(self) -> float:
        self.handle.join()
        return self.queue.get()


class CMAES:
    def __init__(
            self,
            dim: int,
            generation: int,
            population: int,
            mu: int = -1,
            sigma: float = 0.3,
            minimalize: bool = True,
            max_thread: int = 1
    ):
        self._base = BaseCMAES(dim, population, mu, sigma, minimalize, max_thread)
        self._generation = generation

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
            self._base.optimize_current_generation(gen, self._generation, env_creator, _ThreadProc)

    def optimize_with_recoding_min(self, env_creator: MuJoCoEnvCreator, window: Window, camera: Camera):
        for gen in range(1, self._generation + 1):
            good_para = self._base.optimize_current_generation(gen, self._generation, env_creator, _ThreadProc)

            time = datetime.datetime.now()
            recorder = Recorder(f"{gen}({time.strftime('%y%m%d_%H%M%S')}).mp4", 30, 640, 480)
            window.set_recorder(recorder)
            env = env_creator.create_mujoco_env()
            env.calc_and_show(good_para, window, camera)
            window.set_recorder(None)
