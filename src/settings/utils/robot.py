import mujoco
import torch
import numpy as np

from .brain import NeuralNetwork
from .distance_measure import DistanceMeasure
from ..hyper_parameters import HyperParameters


class _Actuator:
    DIF_WHEEL_EV = np.array([1, -1], dtype=np.float64) / 1.41421356237

    def __init__(self, data, i):
        self._act_vector = np.zeros((3, 1), dtype=np.float64)
        self._quat_buf = np.zeros((4, 1), dtype=np.float64)

        self.movement = 0
        self.rotation = 0

        self.act_rot = data.actuator(f"bot{i}.act.rot")
        self.act_move_x = data.actuator(f"bot{i}.act.pos_x")
        self.act_move_y = data.actuator(f"bot{i}.act.pos_y")

    def _calc_ctrl_as_dif_wheels(self, y):
        self.movement = np.linalg.norm(y)
        ey = y / self.movement
        self.movement *= HyperParameters.MOVE_SPEED
        self.rotation = HyperParameters.TURN_SPEED * np.dot(ey, _Actuator.DIF_WHEEL_EV)
        return self.movement, self.rotation

    def _calc_ctrl_as_pattern_movement(self, y):
        yi = np.argmax(y[0:3])
        if yi == 0:
            self.movement = HyperParameters.MOVE_SPEED * y[3]
            self.rotation = 0.0
        elif yi == 1:
            self.movement = 0.0
            self.rotation = HyperParameters.TURN_SPEED * y[3]
        else:
            self.movement = 0.0
            self.rotation = -HyperParameters.TURN_SPEED * y[3]

        return self.movement, self.rotation

    def update(self, y):
        self._calc_ctrl_as_pattern_movement(y)

    def act(self):
        mujoco.mju_axisAngle2Quat(self._quat_buf, [0, 0, 1], self.act_rot.length)
        mujoco.mju_rotVecQuat(self._act_vector, [0, 1, 0], self._quat_buf)
        self.act_rot.ctrl[0] += self.rotation * HyperParameters.TIMESTEP
        self.act_move_x.ctrl[0] += self._act_vector[0] * self.movement * HyperParameters.TIMESTEP
        self.act_move_y.ctrl[0] += self._act_vector[1] * self.movement * HyperParameters.TIMESTEP


class Robot:
    def __init__(
            self,
            model: mujoco.MjModel, data: mujoco.MjData,
            i, brain: NeuralNetwork,
    ):
        self._quat_buf = np.zeros((4, 1), dtype=np.float64)
        self._bot_direction = np.zeros((3, 1), dtype=np.float64)

        self.cam_name = f"bot{i}.camera"

        self._skip_thinking = int(0.1 / HyperParameters.TIMESTEP)
        self._current_thinking_time = self._skip_thinking

        self.brain = brain

        self.bot_id = i
        self.bot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bot{i}.body")
        self.bot_body = data.body(self.bot_body_id)
        self.bot_act = _Actuator(data, i)

        self.sight = np.zeros(1)
        self.brightness_img = np.zeros((1, HyperParameters.NUM_LIDAR + 1))
        self.input_buf = np.zeros((1, 38))

        kernel_size = 16
        self.shrink_filter = torch.nn.Conv1d(1, 1, kernel_size, int(kernel_size * 0.5), bias=False)
        self.shrink_filter.weight.requires_grad_(False)
        self.shrink_filter.weight.copy_(torch.ones(kernel_size, dtype=torch.float32, requires_grad=False) / kernel_size)

    def calc_relative_angle_to(self, pos):
        v = (pos - self.bot_body.xpos)[0:2]
        v /= np.linalg.norm(v)
        theta = np.dot(v, self._bot_direction[0:2, 0])
        return theta

    def _think(self, m: mujoco.MjModel, d: mujoco.MjData, distance_measure: DistanceMeasure):
        mujoco.mju_rotVecQuat(self._bot_direction, [0, 1, 0], self.bot_body.xquat)

        self.sight = distance_measure.measure_with_img(
            m, d, self.bot_body_id, self.bot_body, self._bot_direction
        )
        self.brightness_img = np.dot(
            self.sight[0, :, :], np.array([[0.299], [0.587], [0.114]])
        ).reshape(
            (1, self.sight.shape[1])
        )

        x = self.shrink_filter.forward(
            torch.from_numpy(self.brightness_img).float()
        )
        np.copyto(self.input_buf, x.numpy())
        x /= 255.0
        # x.transpose_(0, 1)
        y = self.brain.forward(x).detach().numpy()

        return y

    def exec(self, m: mujoco.MjModel, d: mujoco.MjData, distance_measure: DistanceMeasure, act=True):
        self._current_thinking_time += 1

        if self._skip_thinking < self._current_thinking_time:
            self._current_thinking_time = 0
            y = self._think(m, d, distance_measure)
            self.bot_act.update(y)

        if act:
            self.bot_act.act()
