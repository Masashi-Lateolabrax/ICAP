import copy
import os

import pickle
import concurrent.futures


class LogFragment:
    def __init__(self, gen, para):
        self.gen = gen
        self.para = copy.deepcopy(para)
        self.data = []

    def add_score(self, data):
        self.data.append(copy.deepcopy(data))


class _Logger:
    _instance = None

    def __new__(cls, dir_path: str, max_workers: int = 4):
        if cls._instance is None:
            cls._instance = super(_Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, dir_path: str, max_workers: int = 1):
        if not self._initialized:
            self.dir_path = dir_path
            os.makedirs(self.dir_path, exist_ok=True)
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            self._initialized = True

            self.gen = -1
            self.paras = []
            self.fragments = []

    def __del__(self):
        if self.executor:
            self.executor.shutdown(wait=True)

    def set_gen(self, gen: int):
        self.gen = gen

    def create_fragment(self, para):
        fragment = LogFragment(self.gen, para)
        self.fragments.append(fragment)
        return fragment

    def save(self, gen):
        fragments = self.fragments
        self.fragments = []
        future = self.executor.submit(self._save, self.dir_path, gen, fragments)
        future.add_done_callback(lambda f: self._handle_exception(f, gen))

    @staticmethod
    def _save(dir_path, gen, fragments: list[LogFragment]):
        log_file: str = os.path.join(dir_path, f"{gen}.pkl")
        with open(log_file, "bw") as f:
            pickle.dump(fragments, f)

    @staticmethod
    def _handle_exception(future, gen):
        if future.exception() is not None:
            print(f"Error occurred in generation {gen}: {future.exception()}")


Logger = _Logger("./results/logs")
