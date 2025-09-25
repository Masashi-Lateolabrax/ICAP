import dataclasses
import datetime
from functools import partial
import argparse
import asyncio
from typing import Optional

import numpy as np
from icecream import ic

import jax
import jax.numpy as jnp
from flax import nnx

from mujoco import mjx

from framework.prelude import *
from framework.cluster import Client

from examples.config import PracticalSimulator, PracticalController

ic.configureOutput(
    prefix=lambda: f'[{datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}] CLIENT| ',
    includeContext=True
)


@dataclasses.dataclass
class Signal:
    stop: bool = False


async def evaluation(
        settings: Settings, receiver: asyncio.Queue, sender: asyncio.Queue, max_batch_size: int
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
    def jit_set_params(sims, params):
        sims = jax.vmap(lambda s, p: s.update(controller=PracticalController(p)))(sims, params)
        sims = jax.vmap(lambda sim: sim.reset(model))(sims)
        return sims

    @partial(nnx.jit, donate_argnames=("sims",))
    def jit_run_batch(sims):
        return jax.vmap(lambda s: s.step_n(model, episode_length))(sims)

    @partial(nnx.jit, donate_argnames=("sims",))
    def jit_run(sim):
        return sim.step_n(model, episode_length)

    while True:
        packet = await receiver.get()  # Receive task from main function
        if isinstance(packet, Signal):
            if packet.stop:
                print("Stopping evaluation routine.")
                break
        if packet is None or not isinstance(packet, TaskContent):
            print("Received invalid task packet.")
            continue

        start_time = datetime.datetime.now(tz=datetime.UTC)

        parameters = packet.parameter
        batch_size = min(parameters.shape[0], max_batch_size)

        if batch_size > 1:
            simulators = jax.tree.map(lambda x: x[:batch_size], base_simulators)
            simulators = jit_set_params(simulators, parameters[:batch_size])
            simulators = jit_run(simulators)
            results = jax.tree.map(lambda x: x.evaluate(), simulators)
            loss = np.array(results["loss"])

        elif batch_size == 1:
            simulator = jit_set_params(base_simulator, parameters[:1])
            simulator = jit_run(simulator)
            results = simulator.evaluate()
            loss = np.array([results["loss"]])

        else:
            raise ValueError("Batch size must be at least 1.")

        end_time = datetime.datetime.now(tz=datetime.UTC)

        result_content = ResultContent(
            result=[(p, l) for p, l in zip(parameters[:batch_size], loss)],
            start_time=start_time,
            end_time=end_time,
        )

        await sender.put(result_content)  # Send result back to main function


async def main():
    parser = argparse.ArgumentParser(description="ICAP Optimization Client")
    parser.add_argument("--host", default="localhost", type=str, help="Server host address")
    parser.add_argument("--port", default=50000, type=int, help="Server port number")
    parser.add_argument("--net-timeout", type=int, default=1, help="TODO")
    parser.add_argument("--heartbeat-interval", type=int, default=5, help="Interval in seconds to send state updates")
    parser.add_argument(
        "--heartbeat-timeout", type=int, default=15, help="Timeout in seconds to detect lost connection"
    )
    args = parser.parse_args()

    host = args.host
    port = args.port
    net_timeout = args.net_timeout
    heartbeat_interval = args.heartbeat_interval
    heartbeat_timeout = args.heartbeat_timeout

    settings = Settings()

    print("=" * 50)
    print("OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {host}:{port}")
    print("-" * 30)
    print("Connecting to server...")
    print("Press Ctrl+C to disconnect")
    print("=" * 50)

    sender = asyncio.Queue()  # Queue for sending tasks TO evaluation function
    receiver = asyncio.Queue()  # Queue for receiving results FROM evaluation function
    task = asyncio.create_task(evaluation(settings, sender, receiver, max_batch_size))

    client = WorkerClient()
    await client.start(host, port, timeout=net_timeout)

    task_content = None
    count = 0
    while count < 5:
        packet: WorkerPacket = ic(await client.receive())

        if packet is None:
            print("Failed to receive packet from server.")
            count += 1
            await asyncio.sleep(retry_interval)
            continue
        count = 0  # Reset counter on successful packet

        if packet.type == WorkerPacketType.TASK:
            if not isinstance(packet.content, TaskContent):
                print("Received invalid packet from server.")
                continue
            if task_content is not None:
                print("Previous task is still being processed. Ignoring new task.")
                continue
            task_content = packet.content
            await sender.put(task_content)  # Send task to evaluation function

        elif packet.type == WorkerPacketType.STATE:
            await client.send_worker_state(
                gpu_usage=float("nan"),
                working=task_content is not None
            )

        if task is not None and not receiver.empty():
            content = await receiver.get()  # Receive result from evaluation function
            if not isinstance(content, ResultContent):
                print("Received invalid result from evaluation.")
                continue
            task_content = None
            await client.send_worker_result(content)

    print("Connection lost after multiple failed attempts.")
    # Stop evaluation task
    await sender.put(Signal(stop=True))
    await task


if __name__ == "__main__":
    asyncio.run(main())
