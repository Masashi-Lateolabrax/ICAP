import colorsys
import numpy
import scipy
import mujoco

import torch


class Pheromone:
    def __init__(
            self,
            width: int, height: int, d: float,
            saturated_vapor: float, evaporate: float, diffusion: float, decrease: float
    ):
        self.liquid = torch.tensor([[[0.] * height] * width])
        self.gas = torch.tensor([[[0.] * height] * width])

        self.d = torch.tensor([d])

        self.saturated_vapor = torch.tensor([saturated_vapor])
        self.evaporate = torch.tensor([evaporate])
        self.diffusion = torch.tensor([diffusion])
        self.decrease = torch.tensor([decrease])

        self._c = torch.tensor([[[
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]]]) * self.diffusion / torch.pow(self.d, 2)

    def _check_index_is_valid(self, x: int, y: int):
        size = self.liquid.size()
        if x < 0 or size[1] < x:
            raise f"'x' must be in [0, {size[1]}]."
        elif y < 0 or size[2] < y:
            raise f"'y' must be in [0, {size[2]}]."

    def get_liquid_value(self, x: int, y: int) -> float:
        self._check_index_is_valid(x, y)
        return self.liquid[0, x, y]

    def get_gas_value(self, x: int, y: int):
        self._check_index_is_valid(x, y)
        return self.gas[0, x, y]

    @staticmethod
    def calc_weight_for_cells(x: float, y: float):
        """
        This method calculate weights for cells around [x, y].
        This use the bi linear method to calculate the weights.
        """

        result = {
            "upper_left": {"index": (0, 0), "weight": 0.0},
            "lower_right": {"index": (0, 0), "weight": 0.0},
            "upper_right": {"index": (0, 0), "weight": 0.0},
            "lower_left": {"index": (0, 0), "weight": 0.0},
        }

        if x - int(x) < 0.5:
            left_index = (int(x) - 1, int(y))
        else:
            left_index = (int(x), int(y))

        if y - int(y) < 0.5:
            upper_index = (int(x), int(y) - 1)
        else:
            upper_index = (int(x), int(y))

        result["upper_left"]["index"] = upper_left_index = (left_index[0], upper_index[1])
        result["lower_right"]["index"] = lower_right_index = (upper_left_index[0] + 1, upper_left_index[1] + 1)
        result["upper_right"]["index"] = (lower_right_index[0], upper_left_index[1])
        result["lower_left"]["index"] = (upper_left_index[0], lower_right_index[1])

        a = x - (upper_left_index[0] + 0.5)
        b = y - (upper_left_index[1] + 0.5)

        result["upper_left"]["weight"] = (1 - a) * (1 - b)
        result["lower_right"]["weight"] = a * b
        result["upper_right"]["weight"] = a * (1 - b)
        result["lower_left"]["weight"] = (1 - a) * b

        return result

    def add_liquid(self, x: int, y: int, value: float):
        self._check_index_is_valid(x, y)
        weights = self.calc_weight_for_cells(x, y)

        for iw in weights.values():
            i = iw["index"]
            w = iw["weight"]
            self.liquid[0, i[0], i[1]] = w * value

    def step(self, dt: float):
        dif_liquid = torch.min(self.liquid, (self.saturated_vapor - self.gas) * self.evaporate * dt)

        dif_gas = torch.nn.functional.conv2d(self.gas, self._c, padding="same")
        dif_gas -= self.gas * self.decrease
        dif_gas *= dt
        dif_gas += dif_liquid

        self.liquid -= dif_liquid
        self.gas += dif_gas

        print(self.liquid)
