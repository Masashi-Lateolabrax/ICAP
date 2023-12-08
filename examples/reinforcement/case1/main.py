import random

import torch.optim


def setup(time_step, replay_buf_size: int, device):
    import mujoco
    from examples.reinforcement.case1 import environment
    from examples.reinforcement.case1 import nn_model

    nn = nn_model.DQN().to(device)
    prev_nn = nn_model.DQN().to(device)
    prev_nn.load_state_dict(nn.state_dict())
    prev_nn.eval()

    optimizer = torch.optim.Adam(
        nn.parameters(), lr=1e-2
    )
    replay_buf = nn_model.ReplayMemory(6, replay_buf_size)

    builder = environment.EnvironmentBuilder(
        [0, 0, 0.5],
        5,
        10000,
        5000,
        1000,
        time_step
    )

    model: mujoco.MjModel = builder.build()
    return nn, prev_nn, optimizer, replay_buf, model, mujoco.MjData(model)


def main():
    import mujoco.viewer
    import torch
    import math

    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_epsilon = 0.9
    end_epsilon = 0.05
    epsilon_decay = 10000
    gamma = 0.99

    time_step = 0.02
    area_size = 20
    life_time = int((60 / time_step) * 1)
    think_interval = 2

    replay_buf_size = int(life_time / think_interval) * 5
    batch_size = 16

    epoc = 0
    action_id = 0
    game_step = 0
    prev_pole_velocity = 0.0
    think_step = think_interval

    loss = torch.Tensor([0.0])
    input_tensor = torch.tensor([0.0] * 6, device=device, requires_grad=False)
    states = torch.tensor([[0.0] * 6] * batch_size, device=device, requires_grad=False)
    actions = torch.tensor([[0]] * batch_size, dtype=torch.int64, device=device, requires_grad=False)
    next_states = torch.tensor([[0.0] * 6] * batch_size, device=device, requires_grad=False)
    targets = torch.tensor([[0.0]] * batch_size, device=device, requires_grad=False)

    nn, prev_nn, optimizer, replay_buf, model, data = setup(time_step, replay_buf_size, device)

    mujoco.mj_step(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            game_step += 1
            think_step += 1
            epsilon = end_epsilon + (start_epsilon - end_epsilon) * math.exp(-1. * epoc / epsilon_decay)

            d = mujoco.mju_norm3(data.site("body_site").xpos)
            a = abs(data.sensor('pole_angle').data[0].item())

            if game_step >= life_time or d > area_size or a > 1.57:
                if d < area_size and a < 1.57:
                    replay_buf.set_reward(game_step)
                else:
                    replay_buf.set_reward(-life_time)
                replay_buf.push_end_episode()

                game_step = 0
                think_step = think_interval
                epoc += 1
                print(f"epoc:{epoc}, epsilon:{epsilon}, loss{loss.item()}")

                mujoco.mj_resetData(model, data)

            if think_step >= think_interval:
                think_step = 0

                replay_buf.set_reward(game_step)

                if replay_buf.is_filled():
                    prev_nn.load_state_dict(nn.state_dict())

                    with torch.no_grad():
                        for i in range(batch_size):
                            replay = replay_buf.select_randomly()
                            states[i, :] = replay["state"]
                            next_states[i, :] = replay["next_state"]
                            actions[i, 0] = replay["action"]
                            targets[i, 0] = replay["reward"]
                        targets[:, 0] += prev_nn.forward(next_states).max(1).values * gamma

                    loss = torch.nn.functional.smooth_l1_loss(
                        nn.forward(states).gather(1, actions),
                        targets
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    for p in nn.parameters():
                        p.grad.data.clamp(-1, 1)
                    optimizer.step()

                if random.random() < epsilon:
                    action_id = random.randrange(0, 17, 1)
                else:
                    input_tensor[0] = d
                    input_tensor[1] = data.sensor('body_velocity').data[0]
                    input_tensor[2] = data.sensor('body_force').data[0] * 0.0001
                    input_tensor[3] = (data.sensor('pole_velocity').data[0] - prev_pole_velocity) / model.opt.timestep
                    input_tensor[4] = data.sensor('pole_velocity').data[0]
                    input_tensor[5] = data.sensor('pole_angle').data[0]

                    q_array = nn.forward(input_tensor)

                    action_id = torch.argmax(q_array).item()

                replay_buf.push(input_tensor, action_id)

            data.actuator("vehicle_act").ctrl = (action_id - 8) * 0.125 * 30

            prev_pole_velocity = data.sensor('pole_velocity').data[0]
            mujoco.mj_step(model, data)

            viewer.sync()


if __name__ == '__main__':
    main()
