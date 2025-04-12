import dataclasses

import numpy as np
import pickle


class Dump:
    @dataclasses.dataclass
    class Delta:
        robot_outputs: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
        robot_pos: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
        food_pos: np.ndarray = np.zeros(0)

    def __init__(self, file_path: str = None):
        self.deltas = []
        if file_path is not None:
            with open(file_path, 'rb') as f:
                self.deltas = pickle.load(f)

        self.deltas.append(Dump.Delta())

    def record_robot_pos(self, name: str, pos: np.ndarray):
        self.deltas[-1].robot_pos[name] = pos.copy()

    def record_robot_outputs(self, name: str, output: np.ndarray):
        self.deltas[-1].robot_outputs[name] = output.copy()

    def record_food_pos(self, food_pos: np.ndarray):
        self.deltas[-1].food_pos = food_pos.copy()

    def step(self):
        self.deltas.append(Dump.Delta())

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.deltas, f)

    def __getitem__(self, item):
        return self.deltas[item]

    def __len__(self):
        return len(self.deltas)

    def __iter__(self):
        return iter(self.deltas)
