import os
import numpy as np
from lib.optimizer import Hist
from environments.collect_feed_without_obstacle import EnvCreator


def set_env_creator(env_creator: EnvCreator):
    print(f"DIMENSION : {env_creator.dim()}")

    env_creator.nest_pos = (0, 0)
    env_creator.robot_pos = [
        (-45, 45), (0, 45), (45, 45),
        (-45, 0), (0, 0), (45, 0),
        (-45, -45), (0, -45), (45, -45),
    ]
    env_creator.obstacle_pos = [(0, 300)]
    env_creator.feed_pos = [(0, 600), (0, 1000)]

    env_creator.pheromone_field_pos = (0, 550)
    env_creator.pheromone_field_panel_size = 20
    env_creator.pheromone_field_shape = (60, 80)

    env_creator.sv = 10.0
    env_creator.evaporate = 20.0
    env_creator.diffusion = 35.0
    env_creator.decrease = 0.1

    env_creator.timestep = int(30 / 0.033333)


def dump(task_dir):
    hist_path = os.path.join(task_dir, "history_77102066.npz")
    hist = Hist.load(hist_path)
    para = hist.get_min().min_para

    env_creator = EnvCreator()
    set_env_creator(env_creator)

    env = env_creator.create(para)

    features = np.zeros((env.timestep, 9, 6))
    outputs = np.zeros((env.timestep, 9, 3))
    sensed_pheromone = np.zeros((env.timestep, 9))
    food_distance = np.zeros((env.timestep, 2))
    pheromone = np.zeros((env.timestep, env.pheromone_field._nx, env.pheromone_field._ny))

    for _ in range(0, 5):
        env.model.step()
    for t in range(0, env.timestep):
        env.calc_step()
        # env.render()

        for i, f in enumerate(env.feeds):
            food_distance[t, i] = np.linalg.norm(f.get_pos() - env.nest_pos)

        for i, bot in enumerate(env.robots):
            features[t, i, :] = bot.brain.get_calced_feature_value()
            outputs[t, i, :] = bot.brain.get_output()
            sensed_pheromone[t, i] = bot.brain.get_input()[6]

        pheromone[t] = env.pheromone_field._gas.T[:, :]

    np.save(os.path.join(task_dir, "features.npy"), features)
    np.save(os.path.join(task_dir, "outputs.npy"), outputs)
    np.save(os.path.join(task_dir, "sensed_pheromone.npy"), sensed_pheromone)
    np.save(os.path.join(task_dir, "food_distance.npy"), food_distance)
    np.save(os.path.join(task_dir, "pheromone.npy"), pheromone)


def main():
    cd = os.path.dirname(__file__)

    for task_dir in os.listdir(os.path.join(cd, "results")):
        task_dir = os.path.join(cd, f"results/{task_dir}")

        if not os.path.isdir(task_dir):
            continue

        hist_path = os.path.join(task_dir, "history_77102066.npz")
        if not os.path.exists(hist_path):
            continue

        print(task_dir)

        # dump(task_dir)
        f_existing = os.path.exists(os.path.join(task_dir, "features.npy"))
        o_existing = os.path.exists(os.path.join(task_dir, "outputs.npy"))
        sp_existing = os.path.exists(os.path.join(task_dir, "sensed_pheromone.npy"))
        d_existing = os.path.exists(os.path.join(task_dir, "food_distance.npy"))
        p_existing = os.path.exists(os.path.join(task_dir, "pheromone.npy"))
        if not f_existing or not o_existing or not sp_existing or not d_existing or not p_existing:
            print("dump")
            dump(task_dir)


if __name__ == '__main__':
    main()
