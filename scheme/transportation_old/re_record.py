import os

from lib.optimizer import Hist
from studyLib import wrap_mjc, miscellaneous
from environments.collect_feed_without_obstacle import EnvCreator


def set_env_creator(env_creator: EnvCreator):
    print(f"DIMENSION : {env_creator.dim()}")

    env_creator.nest_pos = (0, 0)
    env_creator.robot_pos = [
        (-45, 45), (0, 45), (45, 45),
        (-45, 0), (0, 0), (45, 0),
        (-45, -45), (0, -45), (45, -45),
    ]
    env_creator.obstacle_pos = [(0, 300)]
    env_creator.feed_pos = [(0, 600), (0, 1000)]

    env_creator.pheromone_field_pos = (0, 550)
    env_creator.pheromone_field_panel_size = 20
    env_creator.pheromone_field_shape = (60, 80)

    env_creator.sv = 10.0
    env_creator.evaporate = 20.0
    env_creator.diffusion = 35.0
    env_creator.decrease = 0.1

    env_creator.timestep = int(30 / 0.033333)


def main():
    cd = os.path.dirname(__file__)

    width: int = 500
    height: int = 700
    scale: int = 1

    for task_dir in os.listdir(os.path.join(cd, "results")):
        task_dir = os.path.join(cd, f"results/{task_dir}")

        if not os.path.isdir(task_dir):
            continue

        hist_path = os.path.join(task_dir, "history_77102066.npz")
        if not os.path.exists(hist_path):
            continue

        video_path = os.path.join(task_dir, "re_rec.mp4")
        if os.path.exists(video_path):
            continue

        print(task_dir)

        env_creator = EnvCreator()
        set_env_creator(env_creator)

        history = Hist.load(hist_path)

        window = miscellaneous.Window(width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 600, 0), 2500, 90, 90)
        window.set_recorder(miscellaneous.Recorder(
            video_path,
            30, width, height
        ))

        para = history.get_min().min_para
        env = env_creator.create_mujoco_env(para, window, camera)
        score = env.calc_and_show()

        with open(os.path.join(task_dir, "re_score.txt"), "w") as f:
            f.write(f"Result : {score}")


if __name__ == '__main__':
    main()
