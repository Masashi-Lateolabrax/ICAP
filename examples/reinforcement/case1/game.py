import random
import math
import time

import torch
import mujoco
import mujoco.viewer

from examples.reinforcement.case1 import nn_model


def play(
        device: str,
        nn: nn_model.DQN,
        model: mujoco.MjModel,
        data: mujoco.MjData,

        epsilon: float,

        thinking_interval: int,
        lifetime: int,
        area: float,

        replay_buffer: nn_model.ReplayMemory = None,
        viewer: mujoco.viewer = None
):
    input_buffer = torch.tensor([0.0] * 6, device=device, requires_grad=False)

    action_id = 0
    prev_pole_velocity = 0
    prev_thought_time = -thinking_interval
    prev_sync_time = time.time()

    step = 0
    mujoco.mj_step(model, data)
    while step < lifetime:

        if viewer is not None:
            if viewer.is_running():
                duration = prev_sync_time - time.time()
                if model.opt.timestep > duration:
                    time.sleep(model.opt.timestep - duration)
                viewer.sync()
                prev_sync_time = time.time()
            else:
                break

        step += 1

        d = mujoco.mju_norm3(data.site("body_site").xpos)
        a = abs(data.sensor('pole_angle').data[0].item())

        if step >= lifetime or d > area or a >= math.pi * 0.5:
            if replay_buffer is not None:
                if d <= area and a < math.pi * 0.5:
                    replay_buffer.set_reward(step)
                else:
                    replay_buffer.set_reward(-lifetime)
                replay_buffer.push_end_episode()
            break

        if step - prev_thought_time >= thinking_interval:
            prev_thought_time = step

            input_buffer[0] = d
            input_buffer[1] = data.sensor('body_velocity').data[0]
            input_buffer[2] = data.sensor('body_force').data[0] * 0.0001
            input_buffer[3] = (data.sensor('pole_velocity').data[0] - prev_pole_velocity) / model.opt.timestep
            input_buffer[4] = data.sensor('pole_velocity').data[0]
            input_buffer[5] = data.sensor('pole_angle').data[0]

            if random.random() < epsilon:
                action_id = random.randrange(0, 17, 1)
            else:
                q_array = nn.forward(input_buffer)
                action_id = torch.argmax(q_array).item()

            if replay_buffer is not None:
                replay_buffer.set_reward(step)
                replay_buffer.push(input_buffer, action_id)

        data.actuator("vehicle_act").ctrl = (action_id - 8) * 0.125 * 30

        prev_pole_velocity = data.sensor('pole_velocity').data[0]
        mujoco.mj_step(model, data)
