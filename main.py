import numpy
from environments import utils
from environments.collect_feed import EnvCreator
from studyLib import wrap_mjc, miscellaneous

if __name__ == "__main__":
    def main():
        env_creator = EnvCreator()
        env_creator.nest_pos = (0, 0)
        env_creator.robot_pos = [
            (-45, 45), (0, 45), (45, 45),
            (-45, 0), (0, 0), (45, 0),
            (-45, -45), (0, -45), (45, -45),
        ]
        env_creator.obstacle_pos = [(0, 300)]
        env_creator.feed_pos = [(0, 800), (0, 1200)]

        env_creator.pheromone_field_pos = (0, 700)
        env_creator.pheromone_field_panel_size = 20
        env_creator.pheromone_field_shape = (60, 80)

        env_creator.sv = 10.0
        env_creator.evaporate = 20.0
        env_creator.diffusion = 35.0
        env_creator.decrease = 0.1

        generation = 300
        population = 100
        mu = 10
        sigma = 0.3
        centroid = None
        env_creator.timestep = int(30 / 0.033333)

        # Resume
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
        hist.save("history.log")

        width: int = 640
        height: int = 640
        scale: int = 1
        window = miscellaneous.Window(width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 650, 0), 2100, 90, 90)
        # camera = wrap_mjc.Camera((0, 0, 0), 500, 90, 90)
        window.set_recorder(miscellaneous.Recorder("result.mp4", 30, width, height))

        para = numpy.load("best_para.npy")
        env = env_creator.create_mujoco_env(para, window, camera)
        score = env.calc_and_show()
        print(f"Result : {score}")


    main()