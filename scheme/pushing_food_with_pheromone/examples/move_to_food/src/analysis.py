from unittest.mock import Mock

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from libs.utils.data_collector import BaseDataCollector
from libs.optimizer import Hist

from scheme.pushing_food_with_pheromone.lib.objects.robot import Robot, RobotData

from .prerulde import Settings
from .task import TaskGenerator, Task


class MovementStartLog(BaseDataCollector):
    def __init__(self):
        super().__init__()

        self.robot_angle = []
        self.robot_pos = []
        self.food_pos = []

    def get_episode_length(self) -> int:
        return Settings.SIMULATION_TIME_LENGTH

    def pre_record(self, task, time: int):
        pass

    def record(self, task: Task, time: int, evaluation: float):
        self.robot_angle.append(
            np.copy(task.robot.angle)
        )
        self.robot_pos.append(
            np.copy(task.robot.position)
        )
        self.food_pos.append(
            np.copy(task.food.position)
        )


def analyze(log_path):
    logger = Hist.load(log_path)
    para = logger._hist.queues[-1].min_para

    test_robot = Robot(
        brain=Mock(),
        body_=Mock(),
        data=RobotData(Mock(), Mock(), Mock()),
        actuator=Mock(),
        other_robot_sensor=Mock(),
        food_sensor=Mock(),
    )

    start_map = np.zeros((1000, 2))
    for i in range(start_map.shape[0]):
        print(f"{i} / {start_map.shape[0]}")

        generator = TaskGenerator()
        task = generator.generate(para, debug=True)

        collector = MovementStartLog()
        collector.run(task)

        robot_positions = np.array(collector.robot_pos)
        robot_movement = np.linalg.norm(robot_positions[1:] - robot_positions[:-1], axis=1)
        start_time = np.argmax(robot_movement > 0)

        test_robot._data.pos = robot_positions[start_time]
        test_robot._data.angle = collector.robot_angle[start_time]
        test_robot._data._update_local_direction()

        start_map[i, :] = test_robot.calc_relative_position(collector.food_pos[start_time])[:2]

    plt.scatter(start_map[:, 0], start_map[:, 1])
    plt.show()
