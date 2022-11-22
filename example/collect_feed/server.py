import numpy
from environments import utils, back_enemy
from studyLib import wrap_mjc, miscellaneous

if __name__ == "__main__":
    def main():
        nest_pos = (0, 0)
        robot_pos = [
            [(x, 0) for x in range(-8, 9, 4)],  # 5体
            [(x, 0) for x in range(-6, 7, 4)],  # 4体
            [(x, 0) for x in range(-4, 5, 4)],  # 3体
            [(x, 0) for x in range(-2, 3, 4)]  # 2体
        ]
        enemy_pos = [(0, -30)]
        enemy_weight = 4500

        generation = 100
        population = 100
        timestep = int(10 / 0.03333)

        env = back_enemy.Environment(
            nest_pos,
            robot_pos,
            enemy_pos,
            enemy_weight,
            timestep,
        )

        width: int = 640
        height: int = 480
        scale: int = 2
        window = miscellaneous.Window(width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 0, 0), 100, 0, 90)

        # para, hist = utils.cmaes_optimize(generation, population, 3.0, env, True, (window, camera))
        para, hist = utils.cmaes_optimize_server(generation, population, mu, 3.0, env, 52325, True, (window, camera))
        numpy.save("best_para.npy", para)
        hist.save("history.log")

        para = numpy.load("best_para.npy")
        print(
            "Result : ",
            utils.show_mujoco_env(env, para, window, camera)
        )


    main()
