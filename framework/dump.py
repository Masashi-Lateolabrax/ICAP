import dataclasses

import numpy as np
import pickle


class Dump:
    @dataclasses.dataclass
    class Delta:
        robot_inputs: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
        robot_outputs: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
        robot_pos: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
        robot_direction: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
        food_pos: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))

    def __init__(self, file_path: str = None):
        self.deltas: list[Dump.Delta] = []
        if file_path is not None:
            with open(file_path, 'rb') as f:
                self.deltas = pickle.load(f)

    def create_delta(self):
        delta = Dump.Delta()
        self.deltas.append(delta)
        return delta

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.deltas, f)

    def __getitem__(self, item):
        return self.deltas[item]

    def __len__(self):
        return len(self.deltas)

    def __iter__(self):
        return iter(self.deltas)
