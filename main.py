import numpy

from environments import utils
from environments.collect_feed import EnvCreator
from studyLib import miscellaneous, wrap_mjc


def set_env_creator(env_creator: EnvCreator):
    env_creator.timestep = 0.05
    env_creator.time = 120
    env_creator.think_interval = 10

    theta = 0

    env_creator.nest_pos = (0, 0)
    env_creator.robot_pos = [
        (-140, 140, theta), (-70, 140, theta), (0, 140, theta), (70, 140, theta), (140, 140, theta),
        (-140, 70, theta), (-70, 70, theta), (0, 70, theta), (70, 70, theta), (140, 70, theta),
        (-140, 0, theta), (-70, 0, theta), (0, 0, theta), (70, 0, theta), (140, 0, theta),
        (-140, -70, theta), (-70, -70, theta), (0, -70, theta), (70, -70, theta), (140, -70, theta),
        (-140, -140, theta), (-70, -140, theta), (0, -140, theta), (70, -140, theta), (140, -140, theta),
    ]
    # env_creator.robot_pos = [(0, 700, theta)]
    env_creator.obstacle_pos = [(0, 450)]
    env_creator.feed_pos = [(0, 800), (0, 1100)]

    env_creator.pheromone_field_pos = (0, 550)
    env_creator.pheromone_field_panel_size = 20
    env_creator.pheromone_field_shape = (60, 80)
    env_creator.show_pheromone_index = 0
    env_creator.pheromone_iteration = 5

    env_creator.sv = [10.0, 10.0]
    env_creator.evaporate = [1.0 / env_creator.timestep, 1.0 / env_creator.timestep]
    env_creator.diffusion = [1.0 / env_creator.timestep, 1.0 / env_creator.timestep]
    env_creator.decrease = [0.01 / env_creator.timestep, 0.01 / env_creator.timestep]


if __name__ == '__main__':
    def main():
        env_creator = EnvCreator()
        set_env_creator(env_creator)
        print(f"DIMENSION : {env_creator.dim()}")

        # para = numpy.random.rand(env_creator.dim())
        para, hist = utils.cmaes_optimize_server(
            250,
            100,
            30,
            env_creator,
            52325,
            0.3,
            None,
            True
        )
        numpy.save("best_para.npy", para)
        hist.save("history")

        shape = (5, 7)
        dpi = 100
        scale: int = 1

        width = shape[0] * dpi
        height = shape[1] * dpi
        window = miscellaneous.Window(width * scale * 2, height * scale)
        camera = wrap_mjc.Camera((0, 600, 0), 2500, 90, 90)
        window.set_recorder(miscellaneous.Recorder(
            "replay.mp4", int(1.0 / env_creator.timestep), width * 2, height
        ))

        env = env_creator.create_mujoco_env(para, window, camera)

        for _ in range(int(env_creator.time / env_creator.timestep)):
            env.calc_step()

            env.show_pheromone_index = 0
            env.render((0, 0, width * scale, height * scale), False)
            env.show_pheromone_index = 1
            env.render((width * scale, 0, width * scale, height * scale), False)

            window.flush()


    main()
