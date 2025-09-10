import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import numpy as np

from framework.prelude import *

from ..structure.sim_interface import SimulatorForDebugInterface
from ..for_individual.run import run


def collect_loss(
        save_dir: str,
        settings: Settings,
        recorder: IndividualRecorder,
        simulator: type[SimulatorForDebugInterface],
        file_name: str = "loss_history.pkl",
        seeds: tuple[int] = (0,),
        num_threads: int = 1
):
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path):
        print(f"Loss history already exists at {file_path}. Skipping collection.")
        return

    train_loss = np.array([rec.best_fitness for rec in recorder])
    validation_loss = {s: np.zeros(len(train_loss)) for s in seeds}

    parameters = [rec.best_individual for rec in recorder]
    task_size = (len(parameters) + num_threads - 1) // num_threads
    tasks = [parameters[i:i + task_size] for i in range(0, len(parameters), task_size)]

    def worker(task: list[Individual], seeds_: tuple[int]) -> list[dict]:
        results = []
        for param in task:
            generation = param.generation
            param = deepcopy(param)
            for i, seed in enumerate(seeds_):
                param._generation = seed
                simulation = simulator(settings, param, render=False)
                run(settings, simulation)

                results.append({
                    "generation": generation,
                    "seed": seed,
                    "loss": simulation.loss(),
                })
        return results

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, task, seeds) for task in tasks]

        for f in as_completed(futures):
            for r in f.result():
                generation = r["generation"]
                seed = r["seed"]
                loss = r["loss"]
                validation_loss[seed][generation] = loss

    with open(file_path, 'wb') as f:
        pickle.dump({"train": train_loss, "val": validation_loss}, f)
