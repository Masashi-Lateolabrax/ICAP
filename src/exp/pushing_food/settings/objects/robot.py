import mujoco
import torch
import numpy as np

from ._general import BodyObject

from ..brain import NeuralNetwork
from ..hyper_parameters import HyperParameters


class _Actuator:
    def __init__(self, data: mujoco.MjData, bot_id: int):
        self.act_rot = data.actuator(f"bot{bot_id}.act.rot")
        self.act_move_x = data.actuator(f"bot{bot_id}.act.pos_x")
        self.act_move_y = data.actuator(f"bot{bot_id}.act.pos_y")

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
        self.skip_thinking = int(0.1 / HyperParameters.Simulator.TIMESTEP)
        self.current_thinking_time = self.skip_thinking - 1

    def update(self):
        self.current_thinking_time = (self.current_thinking_time + 1) % self.skip_thinking

    def do_think(self) -> bool:
        return self.current_thinking_time == 0


class Robot(BodyObject):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, bot_id: int, brain: NeuralNetwork):
        super().__init__(model, data, f"bot{bot_id}.body")

        self.cam_name = f"bot{bot_id}.camera"

        self._state = _RobotState()
        self._actuator = _Actuator(data, bot_id)
        self.brain = brain

    def calc_direction(self, out: np.ndarray) -> None:
        mujoco.mju_rotVecQuat(out, [0, 1, 0], self.get_body().xquat)

    def exec(self, bot_direction, input_: torch.Tensor, act=True):
        self._state.update()

        if self._state.do_think():
            y = self.brain.forward(input_).detach().numpy()
            self._actuator.update(y)

        if act:
            self._actuator.act(bot_direction)
