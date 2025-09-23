import dataclasses
import datetime
from functools import partial
import argparse
import asyncio

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx

from mujoco import mjx

from framework.prelude import Settings
from framework.cluster import WorkerClient, ResultContent, TaskContent, WorkerPacket, WorkerPacketType

from examples.config import PracticalSimulator, PracticalController


@dataclasses.dataclass
class Signal:
    stop: bool = False


async def evaluation(
        settings: Settings, receiver: asyncio.Queue, sender: asyncio.Queue, max_batch_size: int
):
    dim = PracticalController.dim()
    episode_length = int(settings.Simulation.TIME_LENGTH / settings.Simulation.TIME_STEP)

    mj_model, base_simulators = PracticalSimulator.new(
        settings,
        PracticalController(jnp.zeros(dim)),
        jax.random.PRNGKey(0)
    )
    model = mjx.put_model(mj_model)

    _, base_simulators = jax.lax.scan(
        lambda c, _x: (c, c.reset(model)),
        init=base_simulators,
        xs=jnp.ones((max_batch_size,), dtype=jnp.int32),
    )

    @partial(nnx.jit, donate_argnames=("sims",))
    def jit_set_params(sims, params):
        sims = jax.vmap(lambda s, p: s.update(controller=PracticalController(p)))(sims, params)
        sims = jax.vmap(lambda sim: sim.reset(model))(sims)
        return sims

    @partial(nnx.jit, static_argnames=("n",), donate_argnames=("sims",))
    def jit_run(sims):
        return jax.vmap(lambda s: s.step_n(model, episode_length))(sims)

    while True:
        packet = await receiver.get()
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

        simulators = jax.tree.map(lambda x: x[:batch_size], base_simulators)

        simulators = jit_set_params(simulators, parameters[:batch_size])
        simulators = jit_run(simulators)

        results = jax.tree.map(lambda x: x.evaluate(), simulators)

        loss = np.array(results["loss"])

        end_time = datetime.datetime.now(tz=datetime.UTC)

        result_content = ResultContent(
            result=[(p, l) for p, l in zip(parameters[:batch_size], loss)],
            start_time=start_time,
            end_time=end_time,
        )

        await sender.put(result_content)

async def main():
    parser = argparse.ArgumentParser(description="ICAP Optimization Client")
    parser.add_argument("--host", type=str, help="Server host address")
    parser.add_argument("--port", type=int, help="Server port number")
    parser.add_argument("--max-batch-size", type=int, default=1, help="Number of tasks to evaluate in batch")
    parser.add_argument("--worker-timeout", type=int, default=600, help="TODO")
    parser.add_argument("--net-timeout", type=int, default=5, help="TODO")
    args = parser.parse_args()

    if not args.host:
        print("Error: --host argument is required")
        exit(1)

    if not args.port:
        print("Error: --port argument is required")
        exit(1)

    host = args.host
    port = args.port
    max_batch_size = args.max_batch_size
    worker_timeout = args.worker_timeout
    net_timeout = args.net_timeout

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

    sender = asyncio.Queue()
    receiver = asyncio.Queue()
    task = asyncio.create_task(evaluation(settings, sender, receiver, max_batch_size))

    client = WorkerClient()
    await client.start(host, port, timeout=net_timeout)

    task = None
    count = 0
    while count < 5:
        packet: WorkerPacket = await client.receive(timeout=net_timeout)

        if packet is None or packet.is_timeout():
            count += 1
            print("Failed to receive initial packet from server.")
            continue
        count = 0

        if packet.type == WorkerPacketType.TASK:
            if not isinstance(packet.content, TaskContent):
                print("Received invalid packet from server.")
                continue
            if task is not None:
                print("Previous task is still being processed. Ignoring new task.")
                continue

        if task is not None:
            receiver.get()

            result: list[tuple[np.ndarray, float]] = task.result()
            await client.send_worker_result(result)
            task = None

    connect_to_server(
        settings.Server.HOST,
        settings.Server.PORT,
        evaluation_function=rosenbrock_function,
    )


if __name__ == "__main__":
    main()
