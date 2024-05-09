import mujoco
import torch
import numpy as np

from .brain import NeuralNetwork
from .distance_measure import DistanceMeasure
from ..hyper_parameters import HyperParameters


class Robot:
    def __init__(
            self,
            model: mujoco.MjModel, data: mujoco.MjData,
            i, brain: NeuralNetwork,
    ):
        self._quat_buf = np.zeros((4, 1), dtype=np.float64)
        self._bot_direction = np.zeros((3, 1), dtype=np.float64)
        self._act_vector = np.zeros((3, 1), dtype=np.float64)
        self._dif_wheel_ev = np.array([1, -1], dtype=np.float64) / 1.41421356237

        self.cam_name = f"bot{i}.camera"

        self._skip_thinking = int(0.1 / HyperParameters.TIMESTEP)
        self._current_thinking_time = self._skip_thinking

        self.brain = brain

        self.bot_id = i
        self.bot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bot{i}.body")
        self.bot_body = data.body(self.bot_body_id)
        self.act_rot = data.actuator(f"bot{i}.act.rot")
        self.act_move_x = data.actuator(f"bot{i}.act.pos_x")
        self.act_move_y = data.actuator(f"bot{i}.act.pos_y")

        self.sight = np.zeros(1)
        self.brightness_img = np.zeros((65,))

        self.movement = 0.0
        self.rotation = 0.0

    def calc_direction(self):
        mujoco.mju_rotVecQuat(self._bot_direction, [0, 1, 0], self.bot_body.xquat)
        mujoco.mju_axisAngle2Quat(self._quat_buf, [0, 0, 1], self.act_rot.length)
        mujoco.mju_rotVecQuat(self._act_vector, [0, 1, 0], self._quat_buf)
        return self._bot_direction

    def calc_relative_angle_to(self, pos):
        v = (pos - self.bot_body.xpos)[0:2]
        v /= np.linalg.norm(v)
        theta = np.dot(v, self._bot_direction[0:2, 0])
        return theta

    def _act_as_dif_wheel(self, y):
        movement = np.linalg.norm(y)
        ey = y / movement
        movement *= HyperParameters.MOVE_SPEED
        rotation = HyperParameters.TURN_SPEED * np.dot(ey, self._dif_wheel_ev)
        return movement, rotation

    def _act_as_pattern_movement(self, y):
        yi = torch.argmax(y[0:3])
        if yi == 0:
            movement = HyperParameters.MOVE_SPEED * y[3].item()
            rotation = 0.0
        elif yi == 1:
            movement = 0.0
            rotation = HyperParameters.TURN_SPEED * y[3].item()
        else:
            movement = 0.0
            rotation = -HyperParameters.TURN_SPEED * y[3].item()
        return movement, rotation

    def exec(self, m: mujoco.MjModel, d: mujoco.MjData, distance_measure: DistanceMeasure, act=True):
        self._current_thinking_time += 1
        self.calc_direction()

        if self._skip_thinking < self._current_thinking_time:
            self._current_thinking_time = 0

            self.sight = distance_measure.measure_with_img(
                m, d, self.bot_body_id, self.bot_body, self._bot_direction
            )
            self.brightness_img = np.dot(self.sight[0, :, :], np.array([[0.299], [0.587], [0.114]])).reshape((65,))

            x = torch.from_numpy(self.brightness_img).float()
            x /= 255.0
            # x.transpose_(0, 1)
            y = self.brain.forward(x).detach().numpy()

            self.movement, self.rotation = self._act_as_dif_wheel(y)

        if act:
            self.act_rot.ctrl[0] += self.rotation * HyperParameters.TIMESTEP
            self.act_move_x.ctrl[0] += self._act_vector[0] * self.movement * HyperParameters.TIMESTEP
            self.act_move_y.ctrl[0] += self._act_vector[1] * self.movement * HyperParameters.TIMESTEP
