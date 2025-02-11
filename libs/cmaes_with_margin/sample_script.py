import numpy as np
import cma.optimizer.cmaeswm as cma
import cma.util.weight as weight


def cma_run():
    minimization_problem = True
    dim = 2
    dim_in = dim // 2

    discrete_space = np.tile(np.arange(-10, 11, 1), (dim_in, 1))  # integer variables

    lam = cma.CMAParam.pop_size(dim)
    w_func = weight.CMAWeightWithNegativeWeights(lam, dim, min_problem=minimization_problem)
    margin = 1 / (dim * lam)

    a, b = 1., 3.
    init_m = np.random.rand(dim) * (b - a) + a
    init_sigma = (b - a) / 2

    opt = cma.CMAESwM(dim, discrete_space, w_func, minimization_problem,
                      lam=lam, m=init_m, sigma=init_sigma,
                      margin=margin, restart=-1, minimal_eigenval=1e-30)

    def target_func(x):
        SI = np.array(x)
        SI[:, (dim - dim_in):] = np.round(SI[:, (dim - dim_in):])
        # SI[:, 0:dim_in] = np.round(SI[:, 0:dim_in])
        SI[:, 0] -= 1
        SI[:, 1] -= 1
        evals = (SI ** 2).sum(axis=1)
        return evals

    res = None
    res_evals = None
    for _ in range(10):
        X = opt.sampling_model().sampling(lam)
        res = X_enc = opt.sampling_model().encoding(lam, X)
        res_evals = evals = target_func(X_enc)
        print("eval: ", np.min(evals))
        opt.update(X, evals)

    best_idx = np.argmin(res_evals)
    best_param = res[best_idx]
    print(f'Best parameter: {best_param}')


if __name__ == '__main__':
    cma_run()
