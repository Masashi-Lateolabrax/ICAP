import abc
import random
import math

from lib.optimizer import MjcTaskInterface


class IntervalTimer:
    def __init__(self, interval):
        self._max_energy = interval
        self._energy = 0

    def count(self, x=1):
        self._energy += x
        res = self._energy >= self._max_energy
        if res:
            self._energy -= self._max_energy
        return res


def random_pos(
        width: float,
        height: float,
        margin: float,
        exception_area: list[tuple[float, float, float]]
) -> tuple[float, float] | None:
    cnt_regeneration = 100

    w1 = width * -0.5 + margin
    w2 = width * 0.5 - margin
    h1 = height * -0.5 + margin
    h2 = height * 0.5 - margin

    res = None
    for i in range(cnt_regeneration):
        x = (w2 - w1) * random.random() + w1
        y = (h2 - h1) * random.random() + h1
        contain = any([math.sqrt((x - a[0]) ** 2 + (y - a[1]) ** 2) <= a[2] for a in exception_area])
        if contain:
            continue
        res = x, y
        break
    if res is None:
        raise "Failed to generate a robot position."

    return res


def random_pos_list(
        num: int,
        width: float, height: float,
        margin1: float, margin2: float
) -> list[tuple[float, float]]:
    exception = []
    res = []
    for _ in range(num):
        pos = random_pos(width, height, margin1, exception)
        exception.append((pos[0], pos[1], margin2))
        res.append(pos)
    return res


def is_overlap(new_circle, circles, r):
    for circle in circles:
        dist = math.sqrt((new_circle[0] - circle[0]) ** 2 + (new_circle[1] - circle[1]) ** 2)
        if dist < 2 * r:
            return True
    return False


def generate_circles(n, r, area):
    circles = []
    attempts = 0
    max_attempts = 10

    while len(circles) < n and attempts < max_attempts:
        angle = random.uniform(0, 2 * math.pi)
        direction = random.uniform(0, 360)
        distance = random.uniform(0, area - r)
        new_circle = (distance * math.cos(angle), distance * math.sin(angle), direction)

        if not is_overlap(new_circle, circles, r):
            circles.append(new_circle)
            attempts = 0
            continue
        attempts += 1

    if attempts == max_attempts:
        print(f"Warning: Maximum attempts reached with {len(circles)} circles placed.")

    return circles


class BaseDataCollector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_episode_length(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def _record(self, task, time: int, evaluation: float):
        raise NotImplementedError()

    def run(self, task: MjcTaskInterface):
        from datetime import datetime

        episode = self.get_episode_length()
        time = datetime.now()
        for t in range(episode):
            if (datetime.now() - time).seconds > 1:
                time = datetime.now()
                print(f"{t}/{episode} ({t / episode * 100}%)")

            evaluation = task.calc_step()
            self._record(task, t, evaluation)
