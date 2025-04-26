import os.path
import datetime

from libs import utils
import framework

from brain import BrainBuilder
from settings import Settings
from logger import Logger


def main():
    git_hash = utils.get_head_hash()[0:8]
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S-") + git_hash
    )
    save_dir = "20250421_124432-3be76128"
    os.makedirs(os.path.join(save_dir, "replay"), exist_ok=True)

    video_file_name = "replay.mp4"
    result_file_name = "result.pkl"

    logger = Logger.load(os.path.join(save_dir, result_file_name))
    ind = logger[-1].min_ind

    settings = Settings()
    brain_builder = BrainBuilder(settings)
    task_generator = framework.TaskGenerator(settings, brain_builder)

    task_generator.robot_positions = [
        (1.1262106310606397, -1.6779491283906323, 214.50974959400716)
    ]
    task_generator.food_positions = [
        (-1.810117866406546, 1.2444780212695203)
    ]

    task = task_generator.generate(ind, debug=True)

    ind.dump = framework.entry_points.record(
        settings,
        os.path.join(save_dir, f"replay/{video_file_name}"),
        task,
    )

    for delta in ind.dump.deltas:
        print(delta.robot_inputs)
        print(delta.robot_outputs)


if __name__ == '__main__':
    main()
