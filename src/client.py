"""
Optimization Client

This script connects to an optimization server to evaluate individuals
using CMA-ES optimization in a distributed manner.
"""
import argparse
import logging
import os
import threading
import time
import math
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
from icecream import ic

from framework.prelude import Settings, TaskProgress
from framework.tasks import SharedTaskManager
from framework.tasks.network_manager import NetworkClient

from config import Simulator, Controller

# Configure icecream for distributed system debugging
ic.configureOutput(
    prefix=lambda: f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] CLIENT| ',
    includeContext=True
)

ic.disable()


def initialize_simulators(settings: Settings, batch_size: int) -> Simulator:
    def sim_init() -> Simulator:
        dummy_params = jnp.zeros(Controller.dim())
        controller = Controller(dummy_params)
        mj_model, sim = Simulator.new(settings, controller, jax.random.PRNGKey(0))
        return sim

    simulators = jax.vmap(lambda _: sim_init())(jnp.arange(batch_size))
    return simulators


def reset_simulators(simulator: Simulator, parameter: jax.Array, rngs: jax.Array) -> Simulator:
    reset_simulator = simulator.reset()
    controller = Controller(parameter)
    return reset_simulator.update(controller=controller, rngs_for_relocating_food=rngs)


def client_evaluation(host: str, port: int, settings: Settings, batch_size: int):
    """Main client evaluation loop using NetworkClient"""

    episode_length = math.ceil(settings.Simulation.TIME_LENGTH / settings.Simulation.TIME_STEP)

    shared_task_manager = SharedTaskManager()
    
    print(f"Initialized {batch_size} simulators")
    initial_simulators = initialize_simulators(settings, batch_size)

    max_retries = 3
    retry_delay = 5.0
    consecutive_failures = 0

    print("Entering main evaluation loop...")
    while True:
        print("Connecting to server...")
        try:
            with NetworkClient(host, port, timeout=10) as client:
                print(f"Connected to server at {host}:{port}")
                
                while True:
                    print("Syncing with server...")
                    sync_success = False
                    for retry in range(max_retries):
                        if client.sync(shared_task_manager):
                            sync_success = True
                            consecutive_failures = 0
                            break
                        else:
                            retry_delay_actual = retry_delay * (2 ** retry)  # Exponential backoff
                            logging.warning(f"Failed to sync with server {host}:{port} (attempt {retry + 1}/{max_retries})")
                            logging.warning(f"retrying in {retry_delay_actual}s")
                            time.sleep(retry_delay_actual)

                    if not sync_success:
                        consecutive_failures += 1
                        logging.error(f"Failed to sync after {max_retries} attempts. Consecutive failures: {consecutive_failures}")

                        if consecutive_failures >= 5:
                            logging.error("Too many consecutive failures. Disconnecting from server.")
                            break

                        time.sleep(retry_delay * 2)
                        continue

                    print("Taking tasks from server...")
                    tasks = shared_task_manager.take_task(n=batch_size)
                    if not tasks:
                        time.sleep(1.0)
                        continue

                    print("Preparing simulators...")
                    num_tasks = len(tasks)
                    sub_simulators = jax.tree.map(lambda x: x[:num_tasks], initial_simulators)
                    sub_simulators = jax.vmap(reset_simulators)(
                        sub_simulators,
                        jnp.array([task.parameter for task in tasks]),
                        jnp.array([task.rng_seed for task in tasks])
                    )

                    print("Running simulations...")
                    sub_simulators = jax.vmap(lambda sim: sim.step_n(episode_length))(sub_simulators)
                    losses = jax.vmap(lambda sim: sim.evaluate())(sub_simulators)
                    losses = np.array(losses)

                    print("Storing results back to server...")
                    for i, task in enumerate(tasks):
                        completed_task = task.replace(
                            result=float(losses[i]["loss"]),
                            progress=TaskProgress.COMPLETED
                        )
                        shared_task_manager.set_task(completed_task)
                        print(f"Completed task with fitness: {float(losses[i]['loss']):.4f}")
                        
        except Exception as e:
            logging.error(f"Connection error: {e}")
            time.sleep(retry_delay)
            continue


def main(settings: Settings):
    parser = argparse.ArgumentParser(description="ICAP Optimization Client")
    parser.add_argument("--host", type=str, help="Server host address")
    parser.add_argument("--port", type=int, help="Server port number")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of tasks to evaluate in batch")
    args = parser.parse_args()

    if not args.host:
        print("Error: --host argument is required")
        exit(1)

    if not args.port:
        print("Error: --port argument is required")
        exit(1)

    host = args.host
    port = args.port

    print("=" * 50)
    print("OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {host}:{port}")
    print("-" * 30)
    print(f"Batch size: {args.batch_size}")
    print("-" * 30)
    print("Connecting to server...")
    print("Press Ctrl+C to disconnect")
    print("=" * 50)

    try:
        client_evaluation(host, port, settings, args.batch_size)
    except Exception as e:
        logging.error(f"Failed to connect to server: {e}")
        exit(1)


if __name__ == "__main__":
    main(
        Settings()
    )
