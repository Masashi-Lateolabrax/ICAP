import os
import numpy as np
import concurrent.futures


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

    def __del__(self):
        if self.executor:
            self.executor.shutdown(wait=True)

    def log(self, gen, solutions, food_robot_eva, nest_food_eva):
        future = self.executor.submit(self._save, gen, solutions, food_robot_eva, nest_food_eva)
        future.add_done_callback(lambda f: self._handle_exception(f, gen))

    def _save(self, gen: int, solutions, food_robot_eva, nest_food_eva):
        log_file: str = os.path.join(self.dir_path, f"{gen}.npz")
        np.savez(log_file, solutions=solutions, food_robot_eva=food_robot_eva, nest_food_eva=nest_food_eva)

    def _handle_exception(self, future, gen):
        if future.exception() is not None:
            print(f"Error occurred in generation {gen}: {future.exception()}")


Logger = _Logger("./results/logs")
