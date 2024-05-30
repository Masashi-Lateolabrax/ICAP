import mujoco
import numpy as np

from ..hyper_parameters import HyperParameters


class _Actuator:
    def __init__(self, data):
        self.act_rot = data.actuator(f"bot.act.rot")
        self.act_move_x = data.actuator(f"bot.act.pos_x")
        self.act_move_y = data.actuator(f"bot.act.pos_y")

        self.movement = 0
        self.rotation = 0

    def update(self, y):
        self.movement = (y[0] + y[1]) * 0.5 * HyperParameters.Robot.MOVE_SPEED
        self.rotation = (y[0] - y[1]) * 0.5 * HyperParameters.Robot.TURN_SPEED

    def act(self, bot_direction):
        move_vec = bot_direction * self.movement
        self.act_rot.ctrl[0] = self.rotation
        self.act_move_x.ctrl[0] = move_vec[0, 0]
        self.act_move_y.ctrl[0] = move_vec[1, 0]
        print(f"{self.act_move_y.ctrl[0] - self.act_move_y.velocity[0]}")


class _RobotState:
    def __init__(self):
        self.bot_direction = np.zeros((3, 1), dtype=np.float64)
        self.skip_thinking = int(0.1 / HyperParameters.Simulator.TIMESTEP)
        self.current_thinking_time = self.skip_thinking - 1

    def update(self, bot_body):
        self.current_thinking_time = (self.current_thinking_time + 1) % self.skip_thinking
        mujoco.mju_rotVecQuat(self.bot_direction, [0, 1, 0], bot_body.xquat)

    def do_think(self) -> bool:
        return self.current_thinking_time == 0


class _RobotInfo:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.cam_name = f"bot.camera"
        self.bot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bot.body")
        self.bot_body = data.body(self.bot_body_id)
        self.bot_act = _Actuator(data)


class Robot:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._state = _RobotState()
        self._info = _RobotInfo(model, data)
        self._actuator = _Actuator(data)

    def exec(self, input_, act=True):
        self._state.update(self._info.bot_body)

        if self._state.do_think():
            match input_:
                case 1:
                    y = [1, 1]
                case 2:
                    y = [-1, -1]
                case 3:
                    y = [-1, 1]
                case 4:
                    y = [1, -1]
                case 5:
                    y = [0.7, 1]
                case 6:
                    y = [1, 0.7]
                case _:
                    y = [0, 0]
            self._actuator.update(y)

        if act:
            self._actuator.act(self._state.bot_direction)
