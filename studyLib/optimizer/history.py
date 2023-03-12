import datetime

import numpy


class Queue:
    def __init__(
            self,
            scores_avg: float,
            centroid: numpy.ndarray,
            min_score: float,
            min_para: numpy.ndarray,
            max_score: float,
            max_para: numpy.ndarray,
            sigma: float
    ):
        self.time = datetime.datetime.now()
        self.scores_avg: float = scores_avg
        self.centroid: numpy.ndarray = centroid.copy()
        self.min_score: float = min_score
        self.min_para: numpy.ndarray = min_para.copy()
        self.max_score: float = max_score
        self.max_para: numpy.ndarray = max_para.copy()
        self.sigma: float = sigma


class Hist:
    def __init__(self, dim: int, population: int, mu: int):
        self.queues: list[Queue] = []
        self.dim = dim
        self.population = population
        self.mu = mu
        self.min_index = 0
        self.max_index = 0
        self.min_cmatrix = numpy.zeros(0)
        self.max_cmatrix = numpy.zeros(0)
        self.last_cmatrix = numpy.zeros(0)

    def save(self, file_path: str = None):
        if file_path is None:
            file_path = f"./TMP_HIST.tmp"

        meta = numpy.array([self.dim, self.population, self.mu])

        time: list[str] = []
        centroids = numpy.zeros((len(self.queues), self.dim))
        min_para = numpy.zeros((len(self.queues), self.dim))
        max_para = numpy.zeros((len(self.queues), self.dim))
        score = numpy.zeros((len(self.queues), 3))
        sigmas = numpy.zeros((len(self.queues),))
        for i, q in enumerate(self.queues):
            time.append(q.time.strftime("%Y-%m-%d %H:%M:%S.%f"))
            centroids[i] = q.centroid
            min_para[i] = q.min_para
            max_para[i] = q.max_para
            score[i] = numpy.array([q.scores_avg, q.min_score, q.max_score])
            sigmas[i] = q.sigma

        numpy.savez(
            file_path,
            meta=meta,
            time=time,
            centroids=centroids,
            min_para=min_para,
            max_para=max_para,
            score=score,
            sigmas=sigmas,
            min_index=self.min_index,
            min_cmatrix=self.min_cmatrix,
            max_index=self.max_index,
            max_cmatrix=self.max_cmatrix,
            last_cmatrix=self.last_cmatrix,
        )

    @staticmethod
    def load(file_path: str):
        npz = numpy.load(file_path)

        meta = npz["meta"]
        this = Hist(dim=meta[0], population=meta[1], mu=meta[2])
        this.min_index = npz["min_index"]
        this.min_cmatrix = npz["min_cmatrix"]
        this.max_index = npz["max_index"]
        this.max_cmatrix = npz["max_cmatrix"]
        this.last_cmatrix = npz["last_cmatrix"]

        time = npz["time"]
        centroids = npz["centroids"]
        min_para = npz["min_para"]
        max_para = npz["max_para"]
        score = npz["score"]
        sigmas = npz["sigmas"]
        for t, centroid, min_p, max_p, s, sigma_ in zip(time, centroids, min_para, max_para, score, sigmas):
            q = Queue(s[0], centroid, s[1], min_p, s[2], max_p, sigma_)
            q.time = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
            this.queues.append(q)

        return this

    def add(
            self,
            scores_avg: float,
            centroid: numpy.ndarray,
            min_score: float,
            min_para: numpy.ndarray,
            max_score: float,
            max_para: numpy.ndarray,
            sigma: float,
            cmatrix: numpy.ndarray,
    ):
        self.queues.append(Queue(scores_avg, centroid, min_score, min_para, max_score, max_para, sigma))
        self.last_cmatrix = cmatrix.copy()
        if self.queues[self.min_index].min_score > min_score:
            self.min_index = len(self.queues) - 1
            self.min_cmatrix = self.last_cmatrix
        if self.queues[self.max_index].max_score < max_score:
            self.max_index = len(self.queues) - 1
            self.max_cmatrix = self.last_cmatrix

    def get_min(self) -> Queue:
        return self.queues[self.min_index]

    def get_max(self) -> Queue:
        return self.queues[self.max_index]
