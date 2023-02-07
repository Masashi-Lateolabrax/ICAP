import numpy
from environments import utils
from environments.collect_feed_without_obstacle import EnvCreator
from studyLib import wrap_mjc, miscellaneous
from main import set_env_creator

if __name__ == "__main__":
    def main():
        env_creator = EnvCreator()
        set_env_creator(env_creator)

        width: int = 500
        height: int = 700
        scale: int = 1
        window = miscellaneous.Window(width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 600, 0), 2500, 90, 90)

        para = numpy.load("best_para.npy")
        env = env_creator.create_mujoco_env(para, window, camera)
        score = env.calc_and_show()
        print(f"Result : {score}")


    main()
