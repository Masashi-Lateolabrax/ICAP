import numpy
from environments import utils, back_enemy
from studyLib import wrap_mjc

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
        enemy_weight = 1000

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

        para, hist = utils.cmaes_optimize(generation, population, 0.3, env)
        numpy.save("best_para.npy", para)
        hist.save("history.log")

        camera = wrap_mjc.Camera((0, 0, 0), 100, 0, 90)
        para = numpy.load("best_para.npy")
        print(
            "Result : ",
            utils.show_mujoco_env(env, para, camera, 640, 480, 2)
        )


    main()
