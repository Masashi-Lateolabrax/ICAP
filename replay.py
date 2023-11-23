import numpy

from environments.collect_feed import EnvCreator
from studyLib import miscellaneous, wrap_mjc

from main import set_env_creator
from studyLib.optimizer import Hist

if __name__ == '__main__':
    def main():
        numpy.random.seed(230606)
        env_creator = EnvCreator()
        set_env_creator(env_creator)

        env_creator.time = 60

        hist = Hist.load("history.npz")
        print(f"DIMENSION : {env_creator.dim()}, GENERATION : {len(hist.queues)}, MIN GEN : {hist.min_index}")

        queue = hist.get_min()
        # queue = hist.get_rangking_Nth(1)
        print(f"TARGET GEN : {hist.queues.index(queue)}, TARGET SCORE : {queue.min_score}")

        para = queue.min_para
        # para = numpy.random.normal(loc=0.0, scale=0.3, size=env_creator.dim())

        shape = (6, 6)
        dpi = 100
        scale: int = 1
        gain_width = 1

        width = shape[0] * dpi
        height = shape[1] * dpi
        window = miscellaneous.Window(gain_width * width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 0, 0), 1300, 90, 90)
        window.set_recorder(miscellaneous.Recorder(
            "replay.mp4", int(1.0 / env_creator.timestep), width, height
        ))

        env = env_creator.create_mujoco_env(para, window, camera)

        for _ in range(int(env_creator.time / env_creator.timestep)):
            env.calc_step()

            for pi in range(len(env_creator.sv)):
                env.render((width * scale * pi, 0, width * scale, height * scale), False)

            window.flush()


    main()
