import numpy as np

from environments.collect_feed_without_obstacle import EnvCreator

from lib.optimizer import CMAES, MultiThreadProc
from lib.utils import get_head_hash

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
    env_creator.feed_pos = [(0, 600), (0, 1000)]

    env_creator.pheromone_field_pos = (0, 550)
    env_creator.pheromone_field_panel_size = 20
    env_creator.pheromone_field_shape = (60, 80)

    env_creator.sv = 10.0
    env_creator.evaporate = 20.0
    env_creator.diffusion = 35.0
    env_creator.decrease = 0.1

    env_creator.timestep = int(30 / 0.033333)


if __name__ == "__main__":
    def main():
        env_creator = EnvCreator()
        set_env_creator(env_creator)

        generation = 2
        population = 3
        mu = 1
        sigma = 0.3
        centroid = None

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

        cmaes = CMAES(
            dim=env_creator.dim(),
            generation=generation,
            population=population,
            sigma=sigma,
            mu=mu
        )
        for gen in range(1, 1 + cmaes.get_generation()):
            env_creator = EnvCreator()
            set_env_creator(env_creator)
            cmaes.optimize_current_generation(env_creator, MultiThreadProc)

        history = cmaes.get_history()
        head_hash = get_head_hash()[0:8]
        history.save(f"history_{head_hash}.npz")

        width: int = 500
        height: int = 700
        scale: int = 1
        window = miscellaneous.Window(width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 600, 0), 2500, 90, 90)
        window.set_recorder(miscellaneous.Recorder("result.mp4", 30, width, height))

        para = history.get_min().min_para
        env = env_creator.create_mujoco_env(para, window, camera)
        score = env.calc_and_show()
        print(f"Result : {score}")


    main()
