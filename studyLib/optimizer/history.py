import datetime
import re

best_pat = re.compile(r"best:(.+)")
queue_pat = re.compile(r"\[(.+)] avg:(.+), min:(.+), max:(.+)")


class Queue:
    def __init__(self, avg: float, min_value: float, max_value: float):
        self.time = datetime.datetime.now()
        self.average = avg
        self.min = min_value
        self.max = max_value


class Hist:
    def __init__(self, minimalize: bool = True):
        self.queues = []
        self.best = float("inf")
        self.minimalize = minimalize

    def is_minimalize(self) -> bool:
        return self.minimalize

    def add(self, avg: float, min_value: float, max_value: float) -> bool:
        self.queues.append(Queue(avg, min_value, max_value))

        if self.minimalize and min_value < self.best:
            self.best = min_value
            return True
        elif not self.minimalize and max_value > self.best:
            self.best = max_value
            return True

        return False

    def save(self, file_path: str):
        with open(file_path, mode="w") as f:
            f.write(f"best:{self.best}\n")
            for q in self.queues:
                f.write(f"[{q.time}] avg:{q.average}, min:{q.min}, max:{q.max}\n")

    def load(self, file_path: str):
        with open(file_path, mode="r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    res = re.match(best_pat, line)
                    self.best = float(res.group(1))
                else:
                    res = re.match(queue_pat, line)
                    q = Queue(float(res.group(2)), float(res.group(3)), float(res.group(4)))
                    q.time = datetime.datetime.fromisoformat(res.group(1))
                    self.queues.append(q)
