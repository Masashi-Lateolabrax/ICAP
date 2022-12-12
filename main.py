import numpy
from environments import utils
from environments.collect_feed import EnvCreator
from studyLib import wrap_mjc, miscellaneous


def set_env_creator(env_creator: EnvCreator):
    print(f"DIMENSION : {env_creator.dim()}")

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
    env_creator.show_pheromone_index = 1

    env_creator.sv = [10.0]
    env_creator.evaporate = [5.0]
    env_creator.diffusion = [30.0]
    env_creator.decrease = [0.01]

    env_creator.timestep = int(30 / 0.033333)


if __name__ == "__main__":
    def main():
        env_creator = EnvCreator()
        set_env_creator(env_creator)

        generation = 300
        population = 150  # int(3.0 * numpy.log(env_creator.dim())) * 4
        mu = 10  # int(3.0 * numpy.log(env_creator.dim())) * 2
        sigma = 0.3
        centroid = None
        print(f"population : {population}, parent number : {mu}")

        # Resume
        # from studyLib import optimizer
        # hist = optimizer.Hist(0, 0, 0)
        # hist.load("./TMP_HIST.npz")
        # best_score = float("inf")
        # for q in hist.queues:
        #     if q.min_score < best_score:
        #         centroid = q.min_para.copy()
        #         sigma = q.sigma
        #         best_score = q.min_score

        # Resume
        # centroid = numpy.load("best_para.npy")

        # para, hist = utils.cmaes_optimize(generation, population, mu, sigma, centroid, env_creator, 1, True)
        para, hist = utils.cmaes_optimize_server(generation, population, mu, sigma, centroid, env_creator, 52325, True)
        numpy.save("best_para.npy", para)
        hist.save("history")

        width: int = 500
        height: int = 700
        scale: int = 1
        window = miscellaneous.Window(len(env_creator.sv) * width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 600, 0), 2500, 90, 90)
        window.set_recorder(miscellaneous.Recorder("result.mp4", 30, len(env_creator.sv) * width, height))

        envs = []
        for i in range(0, len(env_creator.sv)):
            env_creator.show_pheromone_index = i
            env = env_creator.create_mujoco_env(para, window, camera)
            envs.append(env)

        timestep = envs[0].timestep
        score = float("nan")
        for t in range(0, timestep):
            for i, env in enumerate(envs):
                score = env.calc_step()
                env.render((width * i, 0, width, height), i == len(envs) - 1)
        print(f"Result : {score}")


    main()
