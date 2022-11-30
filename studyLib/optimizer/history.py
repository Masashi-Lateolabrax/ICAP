import datetime

import numpy


class Queue:
    def __init__(
            self,
            scores_avg: float,
            avg_para: numpy.ndarray,
            min_score: float,
            min_para: numpy.ndarray,
            max_score: float,
            max_para: numpy.ndarray,
            c: numpy.ndarray
    ):
        self.time = datetime.datetime.now()
        self.scores_avg = scores_avg
        self.avg_para = avg_para.copy()
        self.min_score = min_score
        self.min_para = min_para.copy()
        self.max_score = max_score
        self.max_para = max_para.copy()

        self.c_ave = 0.0
        self.c_max = -float("inf")
        self.c_min = float("inf")
        n = 0.0
        for y in range(0, c.shape[0]):
            for x in range(y, c.shape[1]):
                self.c_ave += c[x, y]
                n += 1.0
                if c[x, y] > self.c_max:
                    self.c_max = c[x, y]
                if c[x, y] < self.c_min:
                    self.c_min = c[x, y]
        self.c_ave /= n


class Hist:
    def __init__(self, dim: int, population: int, mu: int):
        self.queues: list[Queue] = []
        self.dim = dim
        self.population = population
        self.mu = mu

    def save(self, file_path: str = None):
        if file_path is None:
            file_path = f"./TMP_HIST"

        meta = numpy.array([self.dim, self.population, self.mu])

        time: list[str] = []
        avg_para = numpy.zeros((len(self.queues), self.dim))
        min_para = numpy.zeros((len(self.queues), self.dim))
        max_para = numpy.zeros((len(self.queues), self.dim))
        score = numpy.zeros((len(self.queues), 3))
        c = numpy.zeros((len(self.queues), 3))
        for i, q in enumerate(self.queues):
            time.append(q.time.strftime("%Y-%m-%d %H:%M:%S.%f"))
            score[i] = numpy.array([q.scores_avg, q.min_score, q.max_score])
            c[i] = numpy.array([q.c_ave, q.c_min, q.c_max])
            avg_para[i] = q.avg_para
            min_para[i] = q.min_para
            max_para[i] = q.max_para

        numpy.savez(
            file_path,
            meta=meta,
            time=time,
            avg_para=avg_para,
            min_para=min_para,
            max_para=max_para,
            score=score,
            c=c,
        )

    def load(self, file_path: str):
        npz = numpy.load(file_path)
        meta = npz["meta"]
        time = npz["time"]
        avg_para = npz["avg_para"]
        min_para = npz["min_para"]
        max_para = npz["max_para"]
        score = npz["score"]
        c = npz["c"]

        self.dim = meta[0]
        self.population = meta[1]
        self.mu = meta[2]

        self.queues.clear()
        for t, avg_p, min_p, max_p, s, c_ in zip(time, avg_para, min_para, max_para, score, c):
            q = Queue(score[0], avg_p, score[1], min_p, score[2], max_p, c_)
            q.time = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")

    def add(
            self,
            scores_avg: float,
            avg_para: numpy.ndarray,
            min_score: float,
            min_para: numpy.ndarray,
            max_score: float,
            max_para: numpy.ndarray,
            c: numpy.ndarray
    ):
        self.queues.append(Queue(scores_avg, avg_para, min_score, min_para, max_score, max_para, c))
