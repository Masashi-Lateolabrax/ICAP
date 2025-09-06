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

from config import Simulator, Controller

# Configure icecream for distributed system debugging
ic.configureOutput(
    prefix=lambda: f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] CLIENT| ',
    includeContext=True
)

ic.disable()


def evaluate_batch(tasks, settings: Settings):
    """Evaluate a batch of tasks using vectorized simulation"""
    # Extract parameters and create batch arrays
    parameters = jnp.array([task.parameter for task in tasks])
    rng_seeds = jnp.array([task.rng_seed for task in tasks])
    rngs = jax.vmap(jax.random.PRNGKey)(rng_seeds)
    
    # Create vectorized simulator initializer
    def sim_initializer(param, rng):
        controller = Controller(param)
        mj_model, sim = Simulator.new(settings, controller, rng)
        return sim
    
    # Initialize batch of simulators
    simulators = jax.vmap(sim_initializer)(parameters, rngs)
    
    # Run simulation steps
    episode_length = math.ceil(settings.Simulation.TIME_LENGTH / settings.Simulation.TIME_STEP)
    simulators = jax.vmap(lambda sim: sim.step_n(episode_length))(simulators)
    
    # Extract losses
    def extract_loss(sim):
        return sim.evaluate()["loss"]
    
    losses = jax.vmap(extract_loss)(simulators)
    return np.array(losses)


def client_evaluation(host: str, port: int, settings: Settings, batch_size: int):
    """Main client evaluation loop using SharedTaskManager"""
    shared_task_manager = SharedTaskManager()

    while True:
        try:
            # Sync with server to get tasks
            if not shared_task_manager.sync(host, port):
                logging.warning(f"Failed to sync with server {host}:{port}")
                time.sleep(5.0)
                continue

            # Take available tasks
            tasks = shared_task_manager.take_task(n=batch_size)
            if not tasks:
                time.sleep(1.0)
                continue

            # Evaluate tasks using vectorized batch evaluation
            try:
                losses = evaluate_batch(tasks, settings)
                
                # Update tasks with results
                for task, loss in zip(tasks, losses):
                    completed_task = task.replace(
                        result=float(loss),
                        progress=TaskProgress.COMPLETED
                    )
                    shared_task_manager.tasks[task.id.content_hash] = completed_task
                    print(f"Completed task with fitness: {float(loss):.4f}")
                    
            except Exception as e:
                logging.error(f"Error evaluating batch: {e}")
                # Mark all tasks as failed
                for task in tasks:
                    failed_task = task.replace(progress=TaskProgress.FAILED)
                    shared_task_manager.tasks[task.id.content_hash] = failed_task

        except KeyboardInterrupt:
            logging.info("Client interrupted by user")
            break
        except Exception as e:
            logging.error(f"Client error: {e}")
            time.sleep(5.0)
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
    from settings import MySettings

    main(
        MySettings()
    )