import numpy
from environments import utils
from environments.collect_feed import EnvCreator
from studyLib import wrap_mjc, miscellaneous

if __name__ == "__main__":
    def main():
        env_creator = EnvCreator()
        env_creator.nest_pos = (0, 0)
        env_creator.robot_pos = [(0, 0), (40, 0), (-40, 0)]
        env_creator.obstacle_pos = [(0, 150)]
        env_creator.feed_pos = [(0, 500)]
        env_creator.sv = 10.0
        env_creator.evaporate = 20.0
        env_creator.diffusion = 35.0
        env_creator.decrease = 0.1

        generation = 300
        population = 100
        mu = 20
        env_creator.timestep = int(60 / 0.033333)

        width: int = 640
        height: int = 480
        scale: int = 2
        window = miscellaneous.Window(width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 350, 0), 1200, 0, 90)

        para, hist = utils.cmaes_optimize_server(generation, population, mu, 0.3, env_creator, 52325, True)
        numpy.save("best_para.npy", para)
        hist.save("history.log")

        para = numpy.load("best_para.npy")
        env = env_creator.create_mujoco_env()
        score = utils.show_mujoco_env(env, para, window, camera)
        print(f"Result : {score}")


    main()
