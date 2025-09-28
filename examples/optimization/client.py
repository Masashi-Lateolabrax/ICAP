import dataclasses
import datetime
from functools import partial
import argparse
import asyncio
from typing import Optional
import threading
import queue

import numpy as np
from icecream import ic

import jax
import jax.numpy as jnp
from flax import nnx

from mujoco import mjx

from framework.prelude import *
from framework.cluster import Client

from examples.config import PracticalSimulator, PracticalController
from framework.utils import force_garbage_collection, monitor_comprehensive_gpu

ic.configureOutput(
    prefix=lambda: f'[{datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}] CLIENT| ',
    includeContext=True
)

ic.disable()


@dataclasses.dataclass
class Signal:
    stop: bool = False


def evaluation(
        settings: Settings, receiver: queue.Queue, sender: queue.Queue, max_batch_size: int, batch_step: int = None
):
    dim = PracticalController.dim()
    episode_length = int(settings.Simulation.TIME_LENGTH / settings.Simulation.TIME_STEP)

    mj_model, base_simulator = PracticalSimulator.new(
        settings,
        PracticalController(jnp.zeros(dim)),
        jax.random.PRNGKey(0)
    )
    model = mjx.put_model(mj_model)

    _, base_simulators = jax.lax.scan(
        lambda c, _x: (c, c.reset(model)),
        init=base_simulator,
        xs=jnp.ones((max_batch_size,), dtype=jnp.int32),
    )

    @partial(nnx.jit, donate_argnames=("sims",))
    def jit_batch_set_params(sims, params):
        sims = jax.vmap(lambda s, p: s.update(controller=PracticalController(p)))(sims, params)
        sims = jax.vmap(lambda sim: sim.reset(model))(sims)
        return sims

    @partial(nnx.jit, donate_argnames=("sims",), static_argnames=("step",))
    def jit_batch_run(sims, step):
        return jax.vmap(lambda s: s.step_n(model, step))(sims)

    @partial(nnx.jit, donate_argnames=("sim",))
    def jit_set_params(sim, params):
        sim = sim.update(controller=PracticalController(params))
        sim = sim.reset(model)
        return sim

    @partial(nnx.jit, donate_argnames=("sim",), static_argnames=("step",))
    def jit_run(sim, step):
        return sim.step_n(model, step)

    while True:
        try:
            packet = receiver.get(timeout=1)  # Receive task from main function
        except queue.Empty:
            continue

        if isinstance(packet, Signal):
            if packet.stop:
                print("Stopping evaluation routine.")
                break
        if packet is None or not isinstance(packet, TaskContent):
            print("Received invalid task packet.")
            continue

        start_time = datetime.datetime.now(tz=datetime.UTC)

        result: list[tuple[np.ndarray, float]] = []
        parameters: np.ndarray = packet.parameter
        while len(result) < parameters.shape[0]:
            batch_size = min(ic(parameters.shape[0] - len(result)), max_batch_size)
            current_parameters = parameters[len(result):len(result) + batch_size, :]

            print(f"Processing batch: {len(result)} -> {len(result) + batch_size} / {parameters.shape[0]}")

            if batch_size > 1:
                simulators = jax.tree.map(lambda x: x[:batch_size], base_simulators)
                simulators = jit_batch_set_params(simulators, current_parameters)

                t = 0
                while t < episode_length:
                    n = episode_length if batch_step is None else min(batch_step, episode_length - t)
                    simulators = jit_batch_run(simulators, n)
                    t += n
                    if batch_step is not None:
                        print(f"\nStep {t}/{episode_length}")
                        monitor_comprehensive_gpu()

                results = jax.vmap(lambda x: x.evaluate())(simulators)
                loss = np.array(results["loss"])

            elif batch_size == 1:
                base_simulator = jit_set_params(base_simulator, current_parameters[0])

                t = 0
                while t < episode_length:
                    n = episode_length if batch_step is None else min(batch_step, episode_length - t)
                    base_simulator = jit_run(base_simulator, n)
                    t += n
                    if batch_step is not None:
                        print(f"\nStep {t}/{episode_length}")
                        monitor_comprehensive_gpu()

                results = base_simulator.evaluate()
                loss = np.array([results["loss"]])

            else:
                raise ValueError("Batch size must be at least 1.")

            result.extend([(p, float(l)) for p, l in zip(current_parameters, loss)])

            force_garbage_collection()

        end_time = datetime.datetime.now(tz=datetime.UTC)
        result_content = ResultContent(
            result=result,
            start_time=start_time,
            end_time=end_time,
        )

        average = np.average([f for _, f in result_content.result])
        speed = 1.0 / (end_time - start_time).total_seconds()
        print(f"Evaluated {parameters.shape[0]} tasks | Avg Fitness: {average:.4f} | Speed: {speed:.2f} tasks/s")

        sender.put(result_content)  # Send result back to main function  # Send result back to main function


async def main():
    parser = argparse.ArgumentParser(description="ICAP Optimization Client")
    parser.add_argument("--host", default="localhost", type=str, help="Server host address")
    parser.add_argument("--port", default=50000, type=int, help="Server port number")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Number of tasks to evaluate in batch")
    parser.add_argument("--net-timeout", type=int, default=1, help="TODO")
    parser.add_argument("--heartbeat-interval", type=int, default=5, help="Interval in seconds to send state updates")
    parser.add_argument(
        "--heartbeat-timeout", type=int, default=15, help="Timeout in seconds to detect lost connection"
    )
    args = parser.parse_args()

    host = args.host
    port = args.port
    max_batch_size = args.max_batch_size
    net_timeout = args.net_timeout
    heartbeat_interval = args.heartbeat_interval
    heartbeat_timeout = args.heartbeat_timeout

    settings = Settings()

    print("=" * 50)
    print("OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {host}:{port}")
    print("-" * 30)
    print(f"Max batch size: {max_batch_size}")
    print("-" * 30)
    print("Connecting to server...")
    print("Press Ctrl+C to disconnect")
    print("=" * 50)

    sender = queue.Queue()  # Queue for sending tasks TO evaluation function
    receiver = queue.Queue()  # Queue for receiving results FROM evaluation function
    evaluation_thread = threading.Thread(target=evaluation, args=(settings, sender, receiver, max_batch_size, 100))
    evaluation_thread.daemon = True
    evaluation_thread.start()

    client = await Client.new(host, port, net_timeout, heartbeat_interval, heartbeat_timeout)
    task_content = None
    last_state_sent_time = datetime.datetime.now(tz=datetime.UTC)

    while True:
        current_time = datetime.datetime.now(tz=datetime.UTC)
        if ic((current_time - last_state_sent_time).total_seconds() >= heartbeat_interval):
            await client.send_state(
                gpu_usage=float("nan"),
                working=task_content is not None
            )
            last_state_sent_time = current_time

        content = receiver.get_nowait() if not receiver.empty() else None

        if content is not None:
            if not isinstance(content, ResultContent):
                print("Received invalid result from evaluation.")
                continue
            task_content = None
            await client.send_result(content)

        if ic(await client.manage()):
            print("Connection lost. Exiting...")
            break

        packet_list: list[ClusterPacket] = ic(client.receive())
        if not packet_list:
            await asyncio.sleep(1)
            continue

        for packet in packet_list:
            if packet.type == ClusterPacketType.TASK:
                if not isinstance(packet.content, TaskContent):
                    raise ValueError("Received invalid task content.")
                if task_content is not None:
                    print("Previous task is still being processed. Rejecting new task.")
                    await client.send_reject(packet.content)  # Reject new task
                    continue
                task_content = packet.content
                sender.put(task_content)  # Send task to evaluation function

    # Stop evaluation task
    print("Shutting down client...")
    sender.put(Signal(stop=True))
    evaluation_thread.join()
    await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
