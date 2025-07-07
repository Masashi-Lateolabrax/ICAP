import dataclasses
import socket
import logging
import threading
import signal
import queue
import time
from typing import Optional, Callable, Tuple

from ..prelude import *
from ..types.communication import Packet, PacketType
from ._connection_utils import send_packet, receive_packet


@dataclasses.dataclass
class CalculationState:
    idle: bool
    throughput: Optional[float] = None
    individuals: Optional[list[Individual]] = None
    error: Optional[str] = None


def _evaluation_worker(
        task_queue: queue.Queue,
        response_queue: queue.Queue,
        evaluation_function: EvaluationFunction,
        stop_event: threading.Event
) -> None:
    while not stop_event.is_set():
        try:
            individuals = task_queue.get(timeout=1.0)
            if individuals is None:
                logging.info("Received poison pill, stopping evaluation worker")
                break
            response_queue.put(
                CalculationState(idle=False, throughput=None)
            )

        except queue.Empty:
            response_queue.put(
                CalculationState(idle=True)
            )
            continue

        logging.debug(f"Evaluation worker received {len(individuals)} individuals")

        start_time = time.time()

        for individual in individuals:
            if stop_event.is_set():
                break

            try:
                individual.timer_start()
                fitness = evaluation_function(individual)
                individual.timer_end()

                individual.set_fitness(fitness)
                individual.set_calculation_state(CalculationState.FINISHED)

                response_queue.put(
                    CalculationState(
                        idle=False,
                        throughput=1 / (individual.get_elapse() + 1e-10)
                    )
                )

                logging.debug(f"Evaluated individual with fitness: {fitness}")

            except Exception as e:
                logging.error(f"Error during evaluation function execution: {e}")
                individual.set_fitness(float('inf'))
                continue

        response_queue.put(
            CalculationState(
                idle=True,
                individuals=individuals,
            )
        )

    logging.info("Evaluation worker stopped")
