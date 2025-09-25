import argparse
import asyncio
import datetime

import numpy as np
from cmaes import CMA
from icecream import ic

from framework.prelude import *
from framework.cluster import Server, LoadBalancer
from examples.config import PracticalController

ic.configureOutput(
    prefix=lambda: f'[{datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}] SERVER| ',
    includeContext=True
)


async def main():
    parser = argparse.ArgumentParser(description="ICAP CMA-ES Controller Optimization")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--max-generations", type=int, default=50)
    parser.add_argument("--network-timeout", type=float, default=5.0)
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--heartbeat-timeout", type=float, default=15.0)
    args = parser.parse_args()

    dim = PracticalController.dim()
    population_size = args.population_size
    network_timeout = args.network_timeout
    heartbeat_interval = args.heartbeat_interval
    heartbeat_timeout = args.heartbeat_timeout

    server = Server(heartbeat_interval, heartbeat_timeout)
    load_balancer = LoadBalancer()
    await server.start(args.host, args.port, network_timeout)

    # Initialize CMA-ES
    cma = CMA(
        mean=np.zeros(dim, dtype=np.float32),
        sigma=0.1,
        population_size=population_size,
    )

    print(f"Starting optimization: dim={dim}, pop={population_size}")

    for generation in range(args.max_generations):
        if cma.should_stop():
            break

        print(f"\nGeneration {generation + 1}")

        # Get candidates and send to workers
        candidates = [cma.ask() for _ in range(population_size)]
        fitness: list[tuple[np.ndarray, float]] = []

        while len(fitness) < population_size:
            dead_ids = ic(await server.manage())
            for dead_id in dead_ids:
                load_balancer.remove(dead_id)

            packets = server.receive()

            for client_id, packet in packets.items():
                if packet.type == ClusterPacketType.RESULT and isinstance(packet.content, ResultContent):
                    result: ResultContent = packet.content

                    if result.rejected is not None:
                        print(f"Client {client_id} rejected the task.")
                        rejected_task = result.rejected
                        for candidate in rejected_task.parameter:
                            candidates.append(candidate)

                    else:
                        duration = (result.end_time - result.start_time).total_seconds()
                        task_count = len(result.result)
                        load_balancer.register_performance(client_id, task_count, duration)
                        fitness.extend(result.result)

            available_ids = set((await server.get_available_clients()).keys())
            if not available_ids:
                await asyncio.sleep(10)
                continue

            # Load balancing
            for i in available_ids:
                load_balancer.register_performance(i)
            task_allocation = ic(load_balancer.calc_balance(available_ids, len(candidates)))

            # Split candidates according to current allocation
            current_batch = {}
            for worker_id, task_count in task_allocation.items():
                current_batch[worker_id] = []
                for _ in range(min(task_count, len(candidates))):
                    current_batch[worker_id].append(candidates.pop(0))

            # Send current batch to workers (batch = list of candidates → 2D array)
            for worker_id, batch in current_batch.items():
                task = TaskContent(np.array(batch))
                await server.send_task(worker_id, task)

        # Update CMA-ES
        cma.tell(fitness)

        best_fitness = min([f for _, f in fitness], default=float("inf"))
        print(f"Best fitness: {best_fitness:.4f}")

    await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
