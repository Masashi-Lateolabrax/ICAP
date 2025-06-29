import math

import numpy as np

from framework.interfaces import SimulatorBackend


class SimpleAnimatedBackend(SimulatorBackend):
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.time = 0.0

    def step(self):
        self.time += 0.05

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        for y in range(img_buf.shape[0]):
            for x in range(img_buf.shape[1]):
                img_buf[y, x] = [
                    int(50 + 50 * math.sin(self.time * 0.5)),  # Red channel
                    int(30 + 30 * math.cos(self.time * 0.3)),  # Green channel
                    int(80 + 40 * math.sin(self.time * 0.7))  # Blue channel
                ]

        # Draw animated square
        square_size = 50
        center_x = self.width // 2 + int(100 * math.sin(self.time))
        center_y = self.height // 2 + int(100 * math.cos(self.time))

        # Color changes with time
        color = [
            int(255 * (0.5 + 0.5 * math.sin(self.time * 2))),
            int(255 * (0.5 + 0.5 * math.cos(self.time * 1.5))),
            int(255 * (0.5 + 0.5 * math.sin(self.time * 3)))
        ]

        img_buf[
        max(0, center_y - square_size // 2):min(self.height, center_y + square_size // 2),
        max(0, center_x - square_size // 2):min(self.width, center_x + square_size // 2)
        ] = color

    def reset(self):
        self.time = 0.0
