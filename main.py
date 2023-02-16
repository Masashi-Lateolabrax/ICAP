import numpy

from environments import utils
from environments.collect_feed import EnvCreator
from studyLib import miscellaneous, wrap_mjc


def set_env_creator(env_creator: EnvCreator):
    env_creator.timestep = 0.033333
    env_creator.time = 30

    env_creator.nest_pos = (0, 0)
    env_creator.robot_pos = [
        (-140, 140, -180), (-70, 140, -180), (0, 140, -180), (70, 140, -180), (140, 140, -180),
        (-140, 70, -180), (-70, 70, -180), (0, 70, -180), (70, 70, -180), (140, 70, -180),
        (-140, 0, -180), (-70, 0, -180), (0, 0, -180), (70, 0, -180), (140, 0, -180),
        (-140, -70, -180), (-70, -70, -180), (0, -70, -180), (70, -70, -180), (140, -70, -180),
        (-140, -140, -180), (-70, -140, -180), (0, -140, -180), (70, -140, -180), (140, -140, -180),
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


if __name__ == '__main__':
    def main():
        env_creator = EnvCreator()
        set_env_creator(env_creator)

        # para = numpy.zeros(dim)
        para, hist = utils.cmaes_optimize_server(
            1000,
            300,
            100,
            env_creator,
            52325,
            0.3,
            None,
            True
        )
        numpy.save("best_para.npy", para)
        hist.save("history")

        shape = (10, 7)
        dpi = 100
        scale: int = 1

        width = shape[0] * dpi
        height = shape[1] * dpi
        window = miscellaneous.Window(width * scale, height * scale)
        camera = wrap_mjc.Camera((600, 600, 0), 2500, 90, 90)
        window.set_recorder(
            miscellaneous.Recorder("result.mp4", int(1.0 / env_creator.timestep), width, height)
        )

        env = env_creator.create_mujoco_env(para, window, camera)
        env.calc_and_show()


    main()
