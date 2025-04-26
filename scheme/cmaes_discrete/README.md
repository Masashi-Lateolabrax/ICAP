# CMAES Discrete

It is difficult for CMA-ES to optimize discrete output functions.
Because outputs of function is not change too much by the small perturbation of parameters.

So we use a simple trick to convert the discrete output to a discrete distribution.
It allows us to sample from the distribution for robots' discrete action and
the action can be changed by the small perturbation of parameters.

This scheme is a simple example that show this trick is valid.
