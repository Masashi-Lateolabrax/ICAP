import numpy as np
import pickle


class Dump:
    def __init__(self):
        self.robot_pos = []
        self.food_pos = []

    def dump(self, robot_pos: np.ndarray, food_pos: np.ndarray):
        self.robot_pos.append(robot_pos)
        self.food_pos.append(food_pos)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            this = pickle.load(f)
        return this
