"""
Optimization Server

This script starts an optimization server that distributes CMA-ES optimization
tasks to connected clients.
"""
import logging
import os
import threading
import datetime
import subprocess
from typing import Optional
import math

from icecream import ic
from framework.prelude import *
from framework.optimization import OptimizationServer, CMAES

from controller import RobotNeuralNetwork
from settings import MySettings

# Configure icecream for distributed system debugging
ic.configureOutput(
    prefix=lambda: f'[{datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] SERVER| ',
    includeContext=True
)

ic.disable()


def get_git_hash() -> str:
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, cwd=os.path.dirname(__file__))
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class Handler:
    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        self._generation: Optional[int] = None
        self._last_call_time: Optional[datetime.datetime] = None
        self._current_time: datetime.datetime = datetime.datetime.now()
        self._save_directory: str = self._make_result_directory()

    def _make_result_directory(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        git_hash = get_git_hash()
        folder_name = f"{timestamp}_{git_hash}"
        save_directory = os.path.join(self.settings.Storage.SAVE_DIRECTORY, folder_name)
        os.makedirs(save_directory, exist_ok=True)
        return save_directory

    def _calc_statistic(self, cmaes: CMAES, individuals: list[Individual]):
        time_diff = self._current_time - self._last_call_time
        speed = time_diff.total_seconds()  # sec/gen
        remaining_generations = cmaes.max_generation - cmaes.generation
        remaining_seconds = datetime.timedelta(seconds=remaining_generations * speed)
        eta = self._current_time + remaining_seconds

        fitness = [i.get_fitness() for i in individuals]
        ave_fitness = sum(fitness) / len(fitness)
        variance = sum([(f - ave_fitness) ** 2 for f in fitness])
        sd = math.sqrt(variance / len(fitness))

        return ave_fitness, sd, len(fitness) / speed, eta

    def _print_info(self, generation, ave_fitness, sd, speed, eta):
        print(
            f"[{self._current_time.strftime("%H:%M:%S")}] "
            f"Generation: {generation} | "
            f"Average: {ave_fitness:.2f} | "
            f"SD: {sd:.2f} | "
            f"Speed: {speed:.2f} ind/sec | "
            f"ETA: {eta} "
        )

    def _save(self, generation, individuals: list[Individual]):
        try:
            filename = f"generation_{generation:04d}.pkl"
            file_path = os.path.join(self._save_directory, filename)

            num_to_save = max(1, self.settings.Storage.TOP_N) if self.settings.Storage.TOP_N > 0 else len(individuals)
            sorted_individuals = sorted(individuals, key=lambda x: x.get_fitness())
            individuals_to_save = sorted_individuals[:num_to_save]

            saved_ind = SavedIndividual(
                generation=generation,
                avg_fitness=sum(ind.get_fitness() for ind in individuals) / len(individuals),
                timestamp=datetime.datetime.now().isoformat(),
                individuals=individuals_to_save
            )
            saved_ind.save(file_path)

            logging.info(f"Saved {len(individuals_to_save)} individuals to {file_path}")

        except Exception as e:
            print(f"Error saving individuals: {e}")

    def run(self, cmaes: CMAES, individuals: list[Individual]):
        self._current_time = datetime.datetime.now()

        ic(self._current_time, cmaes.generation, len(individuals))

        if self._last_call_time is None:
            self._last_call_time = self._current_time
            return
        if self._generation is not None and cmaes.generation == self._generation:
            return

        self._generation = cmaes.generation

        ave_fitness, sd, speed, eta = self._calc_statistic(cmaes, individuals)
        self._print_info(cmaes.generation, ave_fitness, sd, speed, eta)

        if cmaes.should_stop() or self._generation % self.settings.Storage.SAVE_INTERVAL == 0:
            self._save(cmaes.generation, individuals)

        self._last_call_time = self._current_time


def main():
    dim = RobotNeuralNetwork().dim

    settings = MySettings()

    settings.Server.HOST = "0.0.0.0"
    settings.Optimization.DIMENSION = dim

    print("=" * 50)
    print("OPTIMIZATION SERVER")
    print("=" * 50)
    print(f"Host: {settings.Server.HOST}")
    print(f"Port: {settings.Server.PORT}")
    print(f"Socket Backlog: {settings.Server.SOCKET_BACKLOG}")
    print("-" * 30)
    print(f"Problem dimension: {settings.Optimization.DIMENSION}")
    print(f"Initial sigma: {settings.Optimization.SIGMA}")
    print(f"Population size: {settings.Optimization.POPULATION}")
    print("-" * 30)
    print(f"Save individuals: {settings.Storage.SAVE_INDIVIDUALS}")
    if settings.Storage.SAVE_INDIVIDUALS:
        print(f"Save directory: {settings.Storage.SAVE_DIRECTORY}")
        print(f"Save interval: {settings.Storage.SAVE_INTERVAL} generations")
    print("-" * 30)
    print("Waiting for clients to connect...")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

    handler = Handler(settings)
    server = OptimizationServer(
        settings,
        handler=handler.run
    )

    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Server stopped by user")
        print("=" * 50)
    except Exception as e:
        print(f"\nServer error: {e}")

    # from analysis import record, latest_saved_individual_file, get_latest_folder
    #
    # save_dir = get_latest_folder(os.path.join(".", settings.Storage.SAVE_DIRECTORY))
    # saved_individual = SavedIndividual.load(latest_saved_individual_file(save_dir))
    #
    # record(
    #     settings,
    #     saved_individual.best_individual,
    #     os.path.join(save_dir, "movie.mp4")
    # )


if __name__ == "__main__":
    main()
