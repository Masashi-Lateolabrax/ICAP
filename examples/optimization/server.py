import argparse
import asyncio
import uuid
from typing import Dict, Optional

import numpy as np

from framework.cluster import Head, TaskContent, ResultContent


class OptimizationServer:
    def __init__(self, max_workers: int = 10):
        self.head = Head()
        self.max_workers = max_workers
        self.worker_states: Dict[uuid.UUID, bool] = {}  # worker_id -> is_working
        self.pending_tasks = asyncio.Queue()
        self.completed_results = asyncio.Queue()

    async def start(self, host: str, port: int, timeout: float = 5.0):
        """Start the optimization server"""
        await self.head.start(host, port, timeout)
        print(f"Optimization server started on {host}:{port}")

    async def stop(self):
        """Stop the server and cleanup resources"""
        await self.head.stop()
        await self.head.cleanup()

    async def add_optimization_task(self, parameters: np.ndarray):
        """Add optimization parameters to the task queue"""
        task_content = TaskContent(parameters)
        await self.pending_tasks.put(task_content)

    async def get_optimization_result(self) -> Optional[ResultContent]:
        """Get completed optimization results"""
        if self.completed_results.empty():
            return None
        return await self.completed_results.get()

    async def worker_management_loop(self):
        """Main loop for managing workers and distributing tasks"""
        while True:
            try:
                # Get list of connected workers
                worker_ids = await self.head.get_ids()

                # Update worker states
                for worker_id in worker_ids:
                    if worker_id not in self.worker_states:
                        self.worker_states[worker_id] = False
                        print(f"New worker connected: {worker_id}")

                # Remove disconnected workers
                connected_ids = set(worker_ids)
                disconnected = set(self.worker_states.keys()) - connected_ids
                for worker_id in disconnected:
                    del self.worker_states[worker_id]
                    print(f"Worker disconnected: {worker_id}")

                # Request worker states and assign tasks
                for worker_id in worker_ids:
                    await self.head.request_worker_state(worker_id)
                    state = await self.head.get_worker_state(worker_id)

                    if state is not None:
                        is_working = state.working
                        self.worker_states[worker_id] = is_working

                        # Assign task to idle workers
                        if not is_working and not self.pending_tasks.empty():
                            task_content = await self.pending_tasks.get()
                            await self.head.send_worker_task(worker_id, task_content)
                            print(f"Assigned task to worker {worker_id}")

                # Collect results from workers
                for worker_id in worker_ids:
                    result = await self.head.get_worker_result(worker_id)
                    if result is not None:
                        await self.completed_results.put(result)
                        print(f"Received result from worker {worker_id}")

            except Exception as e:
                print(f"Error in worker management: {e}")

            await asyncio.sleep(1.0)  # Check workers every second


async def main():
    parser = argparse.ArgumentParser(description="ICAP Optimization Server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host address")
    parser.add_argument("--port", type=int, required=True, help="Server port number")
    parser.add_argument("--max-workers", type=int, default=10, help="Maximum number of workers")
    parser.add_argument("--timeout", type=float, default=5.0, help="Network timeout")
    args = parser.parse_args()

    server = OptimizationServer(max_workers=args.max_workers)

    print("=" * 50)
    print("OPTIMIZATION SERVER")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Max workers: {args.max_workers}")
    print("-" * 30)
    print("Starting server...")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    try:
        await server.start(args.host, args.port, args.timeout)

        # Start worker management in background
        management_task = asyncio.create_task(server.worker_management_loop())

        # Example: Add some test tasks
        print("Adding example optimization tasks...")
        for i in range(5):
            # Generate random parameters for testing
            params = np.random.randn(10, 5)  # 10 parameter sets, 5 dimensions each
            await server.add_optimization_task(params)
            print(f"Added task {i+1} with {params.shape[0]} parameter sets")

        # Monitor results
        result_count = 0
        while result_count < 5:  # Wait for all 5 test tasks to complete
            result = await server.get_optimization_result()
            if result is not None:
                result_count += 1
                duration = (result.end_time - result.start_time).total_seconds()
                print(f"Task {result_count} completed in {duration:.2f}s")
                print(f"  Results: {len(result.result)} parameter-loss pairs")

            await asyncio.sleep(0.1)

        print("All test tasks completed!")

    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())