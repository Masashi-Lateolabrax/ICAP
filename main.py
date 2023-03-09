import numpy

from environments import utils
from environments.collect_feed import EnvCreator
from studyLib import miscellaneous, wrap_mjc
from studyLib.optimizer import Hist


def set_env_creator(env_creator: EnvCreator):
    env_creator.timestep = 0.01
    env_creator.time = 7
    env_creator.think_interval = 10

    env_creator.nest_pos = (0, 0)
    pos = (0, 0, 100)
    for theta in range(0, 360, int(360 / 8)):
        v_x = numpy.cos(theta / 180 * numpy.pi) * pos[2]
        v_y = numpy.sin(theta / 180 * numpy.pi) * pos[2]
        env_creator.robot_pos.append((pos[0] + v_x, pos[1] + v_y, 180 + theta))

    # env_creator.robot_pos = [
    #     (pos[0] + pos[2] * ix, pos[1] + pos[2] * iy, 0) for iy in range(-1, 2) for ix in range(-1, 2)
    # ]

    # env_creator.robot_pos = [(0, 700, theta)]
    # env_creator.obstacle_pos = [(0, 450)]

    env_creator.feed_pos = [(0, 300)]

    env_creator.pheromone_field_pos = (0, 200)
    env_creator.pheromone_field_panel_size = 20
    env_creator.pheromone_field_shape = (60, 40)
    env_creator.show_pheromone_index = 0
    env_creator.pheromone_iteration = 5

    env_creator.sv = [10.0]
    env_creator.evaporate = [1.0 / env_creator.timestep]
    env_creator.diffusion = [1.0 / env_creator.timestep]
    env_creator.decrease = [0.01 / env_creator.timestep]


if __name__ == '__main__':
    def main():
        env_creator = EnvCreator()
        set_env_creator(env_creator)
        print(f"DIMENSION : {env_creator.dim()}")

        hist = Hist.load("history_ebf772df.npz")
        best = hist.get_min()

        para, hist = utils.cmaes_optimize_server(
            1000,
            700,
            350,
            env_creator,
            52325,
            best.sigma,
            best.centroid,
            hist.cmatrix,
            True
        )
        numpy.save("best_para.npy", para)
        hist.save("history")

        shape = (5, 7)
        dpi = 100
        scale: int = 1
        gain_width = 1

        width = shape[0] * dpi
        height = shape[1] * dpi
        window = miscellaneous.Window(gain_width * width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 600, 0), 2500, 90, 90)
        window.set_recorder(miscellaneous.Recorder(
            "result.mp4", int(1.0 / env_creator.timestep), width, height
        ))

        env = env_creator.create_mujoco_env(para, window, camera)

        for _ in range(int(env_creator.time / env_creator.timestep)):
            env.calc_step()

            for pi in range(len(env_creator.sv)):
                env.render((width * scale * pi, 0, width * scale, height * scale), False)

            window.flush()


    main()
