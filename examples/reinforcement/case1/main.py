import torch
import mujoco
import mujoco.viewer

import random
import math

from examples.reinforcement.case1 import nn_model, optimization, game
from examples.reinforcement.case1 import environment


def main():
    import setting
    import mujoco.viewer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    builder = environment.EnvironmentBuilder(
        [0, 0, 0.5],
        5,
        10000,
        5000,
        1000,
        setting.time_step
    )
    model: mujoco.MjModel = builder.build()

    nn = nn_model.DQN()

    optimization.optimize(
        device=device,
        nn=nn,
        model=model,

        num_of_epoc=setting.num_of_epoc,
        learning_rate=setting.learning_rate,
        gamma=setting.gamma,

        start_eps=setting.start_epsilon,
        end_eps=setting.end_epsilon,

        replay_buffer_size=setting.replay_buf_size,
        batch_size=setting.batch_size,

        thinking_interval=setting.think_interval,
        lifetime=setting.lifetime,
        area=setting.area,
    )

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        game.play(
            device=device,
            nn=nn,
            model=model,
            data=data,

            epsilon=0.0,

            thinking_interval=setting.think_interval,
            lifetime=setting.lifetime,
            area=setting.area,

            viewer=viewer
        )


if __name__ == '__main__':
    main()
