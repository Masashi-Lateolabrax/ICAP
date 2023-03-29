import random
import numpy

from environments.collect_feed import EnvCreator
from studyLib import miscellaneous, wrap_mjc, optimizer


def set_env_creator(env_creator: EnvCreator):
    env_creator.timestep = 0.01
    env_creator.time = 20
    env_creator.think_interval = 10

    env_creator.nest_pos = (0, 0)

    env_creator.feed_pos = [numpy.array([0, 300]), numpy.array([-250, -250])]

    pos = numpy.zeros(2)
    for _ in range(10):
        min_distance = 0
        while min_distance < 40:
            pos = (2.0 * numpy.random.rand(2) - 1.0) * 450

            min_distance = min(
                numpy.linalg.norm(pos - env_creator.feed_pos[0]),
                numpy.linalg.norm(pos - env_creator.feed_pos[1])
            ) + 15

            for p in env_creator.robot_pos:
                d = numpy.linalg.norm(p[:2] - pos)
                min_distance = min(d, min_distance)

        env_creator.robot_pos.append([
            pos[0], pos[1],
            (2.0 * random.random() - 1.0) * 180
        ])

    env_creator.pheromone_field_pos = (0, 0)
    env_creator.pheromone_field_panel_size = 20
    env_creator.pheromone_field_shape = (50, 50)
    env_creator.show_pheromone_index = 0
    env_creator.pheromone_iteration = 5

    env_creator.sv = [10.0]
    env_creator.evaporate = [1.0 / env_creator.timestep]
    env_creator.diffusion = [0.03 / env_creator.timestep]
    env_creator.decrease = [0.0 / env_creator.timestep]


if __name__ == '__main__':
    def main():
        env_creator = EnvCreator()
        print(f"DIMENSION : {env_creator.dim()}")

        set_env_creator(env_creator)

        cmaes = optimizer.ServerCMAES(
            52325,
            env_creator.dim(),
            50,
            300,
            -1,
            0.3,
            None,
            None,
            True,
        )

        for gen in range(1, cmaes.generation):
            set_env_creator(env_creator)
            cmaes.optimize(gen, env_creator)

        para = cmaes.get_best_para()
        hist = cmaes.get_history()

        numpy.save("best_para.npy", para)
        hist.save("history")

        shape = (6, 6)
        dpi = 100
        scale: int = 1
        gain_width = 1

        width = shape[0] * dpi
        height = shape[1] * dpi
        window = miscellaneous.Window(gain_width * width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 0, 0), 1300, 90, 90)
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
