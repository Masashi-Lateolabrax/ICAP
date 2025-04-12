import copy

import numpy as np
import pickle


class Dump:
    def __init__(self, file_path: str = None):
        self.robot_pos = []
        self.robot_outputs = []
        self.food_pos = []

        if file_path is not None:
            with open(file_path, 'rb') as f:
                this = pickle.load(f)
            self.robot_pos = this.robot_pos
            self.robot_outputs = this.robot_outputs
            self.food_pos = this.food_pos

    def dump(self, robot_pos: np.ndarray, robot_outputs: dict[str, np.ndarray], food_pos: np.ndarray):
        self.robot_pos.append(robot_pos.copy())
        self.robot_outputs.append(copy.copy(robot_outputs))
        self.food_pos.append(food_pos.copy())

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
