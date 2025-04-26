import os.path

from .prerulde import Settings

from .brain import Brain
from .task import TaskGenerator


def train(log_path):
    from libs.optimizer import CMAES, Hist

    logger = Hist(os.path.dirname(log_path))
    cmaes = CMAES(
        dim=Brain.get_dim(),
        generation=Settings.CMAES_GENERATION,
        population=Settings.CMAES_POPULATION,
        mu=Settings.CMAES_MU,
        sigma=Settings.CMAES_SIGMA,
        logger=logger
    )
    for gen in range(1, 1 + cmaes.get_generation()):
        task_generator = TaskGenerator()
        cmaes.optimize_current_generation(task_generator)

    logger.save(os.path.basename(log_path))


def train2(log_path):
    import numpy as np
    import datetime
    import libs.cmaes_with_margin.cma.optimizer.cmaeswm as cma
    import libs.cmaes_with_margin.cma.util.weight as weight
    from libs.cmaes_with_margin.history import Hist

    minimization_problem = True
    dim = Brain.get_dim()
    dim_in = dim // 2
    discrete_space = np.tile(np.arange(-10, 11, 1), (dim_in, 1))
    margin = 1 / (dim * Settings.CMAES_POPULATION)

    w_func = weight.CMAWeightWithNegativeWeights(Settings.CMAES_POPULATION, dim, min_problem=minimization_problem)

    opt = cma.CMAESwM(
        dim,
        discrete_space,
        w_func,
        minimization_problem,
        lam=Settings.CMAES_POPULATION,
        m=np.zeros(dim),
        sigma=Settings.CMAES_SIGMA,
        margin=margin,
        restart=-1,
        minimal_eigenval=1e-30
    )

    logger = Hist(os.path.dirname(log_path))

    best = float("inf")
    for gen in range(Settings.CMAES_GENERATION):
        start_time = datetime.datetime.now()
        print(
            f"[{start_time}] start {gen} gen. ({gen}/{Settings.CMAES_GENERATION}={float(gen) / Settings.CMAES_GENERATION * 100.0}%)"
        )

        evals = np.zeros(Settings.CMAES_POPULATION)
        seeds = opt.sampling_model().sampling(Settings.CMAES_POPULATION)
        paras = opt.sampling_model().encoding(Settings.CMAES_POPULATION, seeds)
        task_generator = TaskGenerator()
        for i, para in enumerate(paras):
            task = task_generator.generate(para)
            evals[i] = task.run()

        fin_time = datetime.datetime.now()

        avg = np.mean(evals)
        min_score = np.min(evals)
        max_score = np.max(evals)
        best = min(min_score, best)

        elapse = float((fin_time - start_time).total_seconds())
        spd = Settings.CMAES_POPULATION / elapse
        e = datetime.timedelta(seconds=(Settings.CMAES_GENERATION - gen) * elapse)
        print(f"[{fin_time}] finish {gen} gen. speed[ind/s]:{spd},", end=" ")
        print(f"avg:{avg}, min:{min_score}, max:{max_score}, best:{best}, etr:{e}")

        logger.log(
            avg=avg,
            min_score=min_score,
            max_para=paras[np.argmax(evals)],
            max_score=max_score,
            min_para=paras[np.argmin(evals)],
            individuals=paras,
        )

        opt.update(seeds, evals)

    logger.save(os.path.basename(log_path))
