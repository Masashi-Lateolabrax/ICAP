import argparse
import asyncio
import numpy as np
from cmaes import CMA
from framework.cluster import Head, TaskContent, LoadBalancer


async def main():
    parser = argparse.ArgumentParser(description="ICAP CMA-ES Controller Optimization")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--controller-dim", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--max-generations", type=int, default=50)
    args = parser.parse_args()

    head = Head()
    load_balancer = LoadBalancer()
    await head.start(args.host, args.port, 5.0)

    # Initialize CMA-ES
    cma = CMA(
        mean=np.zeros(args.controller_dim, dtype=np.float32),
        sigma=0.1,
        population_size=args.population_size,
    )

    print(f"Starting optimization: dim={args.controller_dim}, pop={args.population_size}")

    for generation in range(args.max_generations):
        if cma.should_stop():
            break

        print(f"\nGeneration {generation + 1}")

        dead_worker_ids = set(await head.cleanup())
        worker_ids = set(await head.get_ids())
        if not worker_ids:
            print("No workers connected. Waiting...")
            await asyncio.sleep(30)
            continue
        print(f"Found {len(worker_ids)} workers")

        # Get candidates and send to workers
        candidates = [cma.ask() for _ in range(args.population_size)]
        fitness: list[tuple[np.ndarray, float]] = []

        # Load balancing
        for i in worker_ids:
            load_balancer.register_performance(i)
        for i in dead_worker_ids:
            load_balancer.remove(i)

        while len(candidates) > 0:
            task_allocation = load_balancer.calc_balance(len(candidates))

            # Split candidates according to current allocation
            current_batch = {}
            for worker_id, task_count in task_allocation.items():
                current_batch[worker_id] = []
                for _ in range(task_count):
                    if len(candidates) == 0:
                        break
                    current_batch[worker_id].append(candidates.pop(0))
            pass

            # Send current batch to workers
            for worker_id, batch in current_batch.items():
                task = TaskContent(np.array(batch))
                await head.send_worker_task(worker_id, task)

            # Collect results for current batch
            for worker_id in worker_ids:
                result = await head.get_worker_result(worker_id)
                if result:
                    if result.start_time and result.end_time:
                        duration = (result.end_time - result.start_time).total_seconds()
                        task_count = len(result.result)
                        load_balancer.register_performance(worker_id, task_count, duration)
                        fitness.extend(result.result)

        # Update CMA-ES
        cma.tell(fitness)

        best_fitness = min([f for _, f in fitness], default=float("inf"))
        print(f"Best fitness: {best_fitness:.4f}")

    await head.stop()
    await head.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
