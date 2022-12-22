import numpy

from environments import utils
from environments.collect_feed import EnvCreator
from studyLib import wrap_mjc, miscellaneous, optimizer


def set_env_creator(env_creator: EnvCreator):
    env_creator.timestep = 0.033333
    env_creator.time = 5

    env_creator.nest_pos = (0, 0)
    env_creator.robot_pos = [
        (-45, 45), (0, 45), (45, 45),
        (-45, 0), (0, 0), (45, 0),
        (-45, -45), (0, -45), (45, -45),
    ]
    env_creator.obstacle_pos = [(0, 300)]
    env_creator.feed_pos = [(0, 800), (0, 1100)]

    env_creator.pheromone_field_pos = (0, 550)
    env_creator.pheromone_field_panel_size = 20
    env_creator.pheromone_field_shape = (60, 80)
    env_creator.show_pheromone_index = 0
    env_creator.pheromone_iteration = 5

    env_creator.sv = [10.0]
    env_creator.evaporate = [1.0 / env_creator.timestep]
    env_creator.diffusion = [1.0 / env_creator.timestep]
    env_creator.decrease = [0.01 / env_creator.timestep]


def optimize(
        generation: int, population: int,
        mu: int, sigma: float, centroid,
        env_creator: EnvCreator,
        server_client: bool = False,
) -> (numpy.ndarray, optimizer.Hist):
    dim = env_creator.dim()
    print(f"DIMENSION : {dim}")
    print(f"population : {population}({int(3.0 * numpy.log(dim)) * 4})")
    print(f"parent number : {mu}({int(3.0 * numpy.log(dim)) * 2})")

    if len(centroid) < dim:
        centroid = numpy.zeros(dim)

    if server_client:
        para, hist = utils.cmaes_optimize_server(generation, population, mu, sigma, centroid, env_creator, 52325, True)
    else:
        para, hist = utils.cmaes_optimize(generation, population, mu, sigma, centroid, env_creator, 2, True)

    numpy.save("best_para.npy", para)
    hist.save("history")

    return para, hist


def load_best_para_from_hist(hist_path) -> (numpy.ndarray, float):
    hist = optimizer.Hist(0, 0, 0)
    hist.load(hist_path)

    best_score = float("inf")
    para = numpy.zeros(0)
    sigma = 0.0
    for q in hist.queues:
        if q.min_score < best_score:
            para = q.min_para.copy()
            sigma = q.sigma
            best_score = q.min_score

    return para, sigma


if __name__ == "__main__":
    def main():
        para = []
        sigma = 0.3
        env_creator = EnvCreator()
        set_env_creator(env_creator)

        # Resume
        # para = numpy.load("best_para.npy")
        # para, sigma = load_best_para_from_hist("./TMP_HIST.log.npz")

        para, hist = optimize(
            generation=1000,
            population=100,
            mu=30,
            sigma=sigma,
            centroid=para,
            env_creator=env_creator,
            server_client=True
        )

        width: int = 500
        height: int = 700
        scale: int = 1
        window = miscellaneous.Window(len(env_creator.sv) * width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 600, 0), 2500, 90, 90)
        window.set_recorder(
            miscellaneous.Recorder("result.mp4", int(1.0 / env_creator.timestep), len(env_creator.sv) * width, height)
        )

        env = env_creator.create_mujoco_env(para, window, camera)
        score = env.calc_and_show()
        print(f"Result : {score}")


    main()
