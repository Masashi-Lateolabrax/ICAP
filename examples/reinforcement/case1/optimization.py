import math

import torch
import mujoco

from examples.reinforcement.case1 import nn_model, game


def optimize(
        nn: nn_model.DQN,
        model: mujoco.MjModel,
        device: str,
        num_of_epoc: int,
        batch_size: int,
        replay_buffer_size: int,
        start_eps: int,
        end_eps: int,
        lifetime: int,
        thinking_interval: int,
        area: int,
        gamma: float,
        learning_rate: float,
):
    from examples.reinforcement.case1 import nn_model

    states = torch.tensor([[0.0] * 6] * batch_size, device=device, requires_grad=False)
    actions = torch.tensor([[0]] * batch_size, dtype=torch.int64, device=device, requires_grad=False)
    next_states = torch.tensor([[0.0] * 6] * batch_size, device=device, requires_grad=False)
    targets = torch.tensor([[0.0]] * batch_size, device=device, requires_grad=False)
    loss = torch.tensor([0.0], requires_grad=False)

    prev_nn = nn.__class__().to(device)
    prev_nn.load_state_dict(
        nn.state_dict()
    )
    prev_nn.eval()

    optimizer = torch.optim.Adam(
        nn.parameters(), lr=learning_rate
    )

    replay_buffer = nn_model.ReplayMemory(6, replay_buffer_size)

    for epoc in range(num_of_epoc):
        print(f"[{epoc} / {num_of_epoc}] {loss.item()}")

        game.play(
            device=device,
            nn=nn,
            model=model,
            data=mujoco.MjData(model),

            epsilon=end_eps + (start_eps - end_eps) * math.exp(-1. * epoc / num_of_epoc),

            thinking_interval=thinking_interval,
            lifetime=lifetime,
            area=area,

            replay_buffer=replay_buffer,
        )

        with torch.no_grad():
            prev_nn.load_state_dict(
                nn.state_dict()
            )

            for i in range(batch_size):
                replay = replay_buffer.select_randomly()
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
