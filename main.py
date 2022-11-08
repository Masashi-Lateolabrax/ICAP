import glfw
import numpy
from studyLib import optimizer, miscellaneous, wrap_mjc
from environments.back_enemy import Environment


def show(
        para: numpy.ndarray,
        nest_pos: (float, float),
        robot_pos: list[list[(float, float)]],
        feed_pos: list[(float, float)],
        feed_weight: float,
        timestep: int,
        camera: wrap_mjc.Camera
):
    width = 640
    height = 480
    scale = 2

    window = miscellaneous.Window(width * scale, height * scale)
    window.set_recorder(miscellaneous.Recorder("result.mp4", 30, width, height))

    env = Environment(
        nest_pos,
        robot_pos,
        feed_pos,
        feed_weight,
        timestep,
        camera,
        window
    )
    return env.calc(para)


def optimize(
        generation: int,
        population: int,
        nest_pos: (float, float),
        robot_pos: list[list[(float, float)]],
        feed_pos: list[(float, float)],
        feed_weight: float,
        timestep: int
) -> (numpy.ndarray, optimizer.Hist):
    env = Environment(
        nest_pos,
        robot_pos,
        feed_pos,
        feed_weight,
        timestep,
        # wrap_mjc.Camera((0, 0, 0), 120, 0, 90),
        # miscellaneous.Window(1200, 800)
    )
    # opt = optimizer.CMAES(env, generation, population, 0.3, True)
    # opt.optimize()
    opt = optimizer.ServerCMAES(env, generation, population, 0.3, True)
    opt.optimize(52325)
    return opt.get_best_para(), opt.get_history()


if __name__ == "__main__":
    def main():
        step_size = 0.03333

        generation = 3
        population = 100
        timestep = int(10 / step_size)

        nest_pos = (0, 0)
        robot_pos = [
            [(x, 0) for x in range(-8, 9, 4)],  # 5体
            [(x, 0) for x in range(-6, 7, 4)],  # 4体
            [(x, 0) for x in range(-4, 5, 4)],  # 3体
            [(x, 0) for x in range(-2, 3, 4)]  # 2体
        ]
        feed_pos = [(0, -30)]
        feed_weight = 10000

        if generation is None:
            speed = 0.48040868405184983
            time_min = 60 * 16
            generation = int(time_min * 60 * speed / population)

        # Optimize and Save The Best Parameter of Brain.
        para, hist = optimize(generation, population, nest_pos, robot_pos, feed_pos, feed_weight, timestep)
        numpy.save("best_para.npy", para)
        hist.save("history.log")

        # Replay The Best Agent Behavior
        para = numpy.load("best_para.npy")
        timestep = int(10 / step_size)
        camera = wrap_mjc.Camera((0, 0, 0), 120, 0, 90)
        print(
            "Result : ",
            show(para, nest_pos, robot_pos, feed_pos, feed_weight, timestep, camera)
        )


    main()
