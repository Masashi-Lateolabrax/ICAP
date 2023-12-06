import random

import torch.optim


def setup(replay_buf_size: int):
    import mujoco
    from examples.reinforcement.case1 import environment
    from examples.reinforcement.case1 import nn_model

    nn = nn_model.DQN()

    target_nn = nn_model.DQN()
    target_nn.load_state_dict(nn.state_dict())
    target_nn.eval()

    optimizer = torch.optim.Adam(
        nn.parameters()
    )
    replay_buf = nn_model.ReplayMemory(5, replay_buf_size)

    builder = environment.EnvironmentBuilder(
        [0, 0, 0.5],
        5,
        30000,
        5000,
        100,
        0.01
    )

    model: mujoco.MjModel = builder.build()
    return nn, target_nn, optimizer, replay_buf, model, mujoco.MjData(model)


def main():
    import mujoco.viewer
    import torch

    epsilon = 0.8
    gamma = 0.99

    num_of_updates = 10
    life_time_step = 6000
    think_interval = 10
    replay_buf_size = int(life_time_step / think_interval) * 10

    input_tensor = torch.Tensor([0.0] * 5)
    epoc = 0
    action_id = 0
    game_step = 0
    think_step = think_interval

    nn, target_nn, optimizer, replay_buf, model, data = setup(replay_buf_size)

    mujoco.mj_step(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            game_step += 1
            think_step += 1

            if game_step >= life_time_step:
                mujoco.mj_resetData(model, data)
                replay_buf.push_end_episode()
                game_step = 0
                epoc += 1
                print(epoc)

            if think_step >= think_interval:
                think_step = 0

                input_tensor[0] = data.sensor('body_velocity').data[0]
                input_tensor[1] = data.sensor('body_force').data[0]
                input_tensor[2] = (data.sensor('pole_velocity').data[0] - input_tensor[3]) / model.opt.timestep
                input_tensor[3] = data.sensor('pole_velocity').data[0]
                input_tensor[4] = data.sensor('pole_angle').data[0]

                q_array = nn.forward(input_tensor)

                if random.random() < epsilon:
                    action_id = random.randrange(0, 17, 1)
                else:
                    action_id = torch.argmax(q_array).item()

                mujoco.mj_step(model, data)

                replay_buf.push(
                    input_tensor,
                    abs(input_tensor[4].item()),
                    torch.max(q_array).item()
                )

                if replay_buf.is_filled():
                    for _ in range(num_of_updates):
                        replay = replay_buf.select_randomly()

                        target = replay["reward"] + gamma * replay["next_max_q"]
                        loss = torch.nn.functional.smooth_l1_loss(
                            nn.forward(replay["state"]),
                            torch.Tensor([target] * 17)
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        for p in nn.parameters():
                            p.grad.data.clamp(-1, 1)
                        optimizer.step()

            data.actuator("vehicle_act").ctrl = (action_id - 8) * 1.25

            viewer.sync()


if __name__ == '__main__':
    main()
