import mujoco
import torch
import numpy as np
import math
import collections

from ._general import BodyObject

from ..brain import NeuralNetwork
from ..hyper_parameters import HyperParameters, StaticParameters


class _Actuator:
    def __init__(self, data: mujoco.MjData):
        self.act_rot = data.actuator(f"bot.act.rot")
        self.act_move_x = data.actuator(f"bot.act.pos_x")
        self.act_move_y = data.actuator(f"bot.act.pos_y")

        self.movement = 0
        self.rotation = 0

    def update(self, y: np.ndarray):
        self.movement = (y[0] + y[1]) * 0.5 * HyperParameters.Robot.MOVE_SPEED
        self.rotation = (y[0] - y[1]) * 0.5 * HyperParameters.Robot.TURN_SPEED

    def act(self, bot_direction):
        move_vec = bot_direction * self.movement
        self.act_rot.ctrl[0] = self.rotation
        self.act_move_x.ctrl[0] = move_vec[0, 0]
        self.act_move_y.ctrl[0] = move_vec[1, 0]


class _RobotState:
    def __init__(self):
        kernel_size = HyperParameters.Robot.SIGHT_KERNEL_SIZE

        self.skip_thinking = int(0.1 / HyperParameters.Simulator.TIMESTEP)
        self.current_thinking_time = self.skip_thinking - 1

        self.sight: np.ndarray = np.zeros((1, HyperParameters.Robot.NUM_LIDAR, 3), dtype=np.uint8)
        self.brightness_img: torch.Tensor = torch.zeros((1, self.sight.shape[1]))
        self.input_buf: torch.Tensor = torch.zeros((1, StaticParameters.input_size()))

        def normal_distribution(x) -> float:
            mean = kernel_size * 0.5
            sigma = kernel_size * 0.5 / 4
            variance_2 = 2 * math.pow(sigma, 2)
            return math.exp(-math.pow(x - mean, 2) / variance_2) / math.sqrt(math.pi * variance_2)

        self._preparation = torch.nn.Sequential(collections.OrderedDict([
            ("padding", torch.nn.CircularPad1d(int(kernel_size * 0.5 + 0.5))),
            ("convolve", torch.nn.Conv1d(1, 1, kernel_size, int(kernel_size * 0.5), bias=False)),
        ]))
        self._preparation.requires_grad_(False)
        self._preparation.convolve.weight.copy_(torch.tensor(
            [normal_distribution(x) for x in range(0, kernel_size)],
            dtype=torch.float32, requires_grad=False
        ))

        self.debug = False

    def update(self, sight: np.ndarray, brightness_img: np.ndarray):
        self.current_thinking_time = (self.current_thinking_time + 1) % self.skip_thinking
        if self.debug:
            np.copyto(self.sight, sight)
            self.brightness_img.copy_(torch.from_numpy(brightness_img))

    def calc_input(self):
        self.input_buf = self._preparation.forward(self.brightness_img)
        self.input_buf /= 255.0

    def do_think(self) -> bool:
        return self.current_thinking_time == 0


class Robot(BodyObject):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, brain: NeuralNetwork):
        super().__init__(model, data, "bot.body")
        self.cam_name = "bot.camera"
        self._actuator = _Actuator(data)

        self._state = _RobotState()
        self.brain = brain

    def calc_direction(self, out: np.ndarray) -> None:
        mujoco.mju_rotVecQuat(out, [0, 1, 0], self.get_body().xquat)

    def get_brightness_buf(self):
        return self._state.brightness_img.detach().numpy().astype(np.uint8)

    def get_input_buf(self):
        buf = (self._state.input_buf.detach().numpy() * 255.0).astype(np.uint8)
        return buf

    def set_debug(self, mode):
        self._state.debug = mode

    def get_debug(self) -> bool:
        return self._state.debug

    def exec(self, bot_direction, sight: np.ndarray, brightness_img: np.ndarray, act=True):
        self._state.update(sight, brightness_img)

        if self._state.do_think():
            self._state.calc_input()
            y = self.brain.forward(self._state.input_buf).detach().numpy()
            self._actuator.update(y)

        if act:
            self._actuator.act(bot_direction)
