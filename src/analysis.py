import os.path
import pickle
import hashlib

import numpy as np

from framework.prelude import *

from client import Simulator
import analysis_mod
from src.analysis_mod import DebugData


class SimulatorForDebugging(analysis_mod.SimulatorForDebugInterface):
    def __init__(self, settings: Settings, parameters: Individual, render: bool = False):
        self.simulator = Simulator(settings, parameters, render)

        self.timestep = settings.Simulation.TIME_STEP
        self.time = 0
        self._debug_data = []

    def step(self):
        self.simulator.step()

        self.time += self.timestep
        self._debug_data.append(analysis_mod.DebugData(
            time=self.time,
            robot_positions=[np.copy(r.xpos) for r in self.simulator.robot_values],
            robot_inputs=self.simulator.input_ndarray.copy(),
            robot_outputs=self.simulator.output_ndarray.copy(),
            robot_directions=[np.copy(r.xdirection) for r in self.simulator.robot_values],
            food_positions=[np.copy(f.xpos) for f in self.simulator.food_values],
            food_directions=[np.copy(f.direction) for f in self.simulator.food_values],
            total_gas_pheromone=self.simulator.get_max_gas_pheromone(),
            total_liquid_pheromone=self.simulator.get_total_liquid_pheromone()
        ))

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        self.simulator.render(img_buf, pos, lookat)

    def debug_data(self) -> list[DebugData]:
        return self._debug_data

    def loss(self):
        return self.simulator.calc_total_score()


def analyze_specific_individual(save_dir: str, settings: Settings, individual: Individual, seed: int = None):
    individual_hash = hashlib.md5(individual.view().tobytes()).hexdigest()[:8]
    generation = individual.generation
    seed = individual.generation if seed is None else seed

    save_dir = os.path.join(save_dir, f"analysis_{generation}_{individual_hash}_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    debug_data_path = os.path.join(save_dir, "debug_data.pkl")
    video_file_path = os.path.join(save_dir, "video.mp4")
    input_anime_file_path = os.path.join(save_dir, "input_anime.mp4")

    robot_sensor_power_file_path = os.path.join(save_dir, "robot_sensor_power.png")
    food_sensor_power_file_path = os.path.join(save_dir, "food_sensor_power.png")
    pheromone_sensor_power_file_path = os.path.join(save_dir, "pheromone_sensor_power.png")

    left_wheel_act_file_path = os.path.join(save_dir, "left_wheel_act.png")
    right_wheel_act_file_path = os.path.join(save_dir, "right_wheel_act.png")
    pheromone_act_file_path = os.path.join(save_dir, "pheromone_act.png")

    total_gas_pheromone_graph_path = os.path.join(save_dir, "total_gas_pheromone.png")
    total_liquid_pheromone_graph_path = os.path.join(save_dir, "total_liquid_pheromone.png")

    # Record the video if not already recorded
    if not os.path.exists(video_file_path):
        # Record the video
        individual._generation = seed
        simulator = SimulatorForDebugging(settings, individual, render=True)
        debug_data = analysis_mod.record(settings, simulator, video_file_path)
        individual._generation = generation

        if not os.path.exists(debug_data_path):
            # Save the debug info
            with open(debug_data_path, 'wb') as f:
                pickle.dump(debug_data, f)

    # Save the debug info if not already saved
    if not os.path.exists(debug_data_path):
        individual._generation = seed
        simulator = SimulatorForDebugging(settings, individual, render=False)
        debug_data = analysis_mod.run(settings, simulator)
        individual._generation = generation

        # Save the debug info
        with open(debug_data_path, 'wb') as f:
            pickle.dump(debug_data, f)

    # Create the input animation if not already created
    if not os.path.exists(input_anime_file_path):
        with open(debug_data_path, 'rb') as f:
            debug_data: list[DebugData] = pickle.load(f)

        analysis_mod.input_animation(settings, debug_data, input_anime_file_path)

    # Plot the robot sensor power
    if not os.path.exists(robot_sensor_power_file_path):
        with open(debug_data_path, 'rb') as f:
            debug_data: list[DebugData] = pickle.load(f)

        analysis_mod.plot_robot_sensor(settings, debug_data, robot_sensor_power_file_path)

    # Plot the food sensor power
    if not os.path.exists(food_sensor_power_file_path):
        with open(debug_data_path, 'rb') as f:
            debug_data: list[DebugData] = pickle.load(f)

        analysis_mod.plot_food_sensor(settings, debug_data, food_sensor_power_file_path)

    # Plot the pheromone sensor power
    if not os.path.exists(pheromone_sensor_power_file_path):
        with open(debug_data_path, 'rb') as f:
            debug_data: list[DebugData] = pickle.load(f)

        analysis_mod.plot_pheromone_sensor(settings, debug_data, pheromone_sensor_power_file_path)

    # Plot the left wheel activity
    if not os.path.exists(left_wheel_act_file_path):
        with open(debug_data_path, 'rb') as f:
            debug_data: list[DebugData] = pickle.load(f)

        analysis_mod.plot_left_wheel_act(settings, debug_data, left_wheel_act_file_path)

    # Plot the right wheel activity
    if not os.path.exists(right_wheel_act_file_path):
        with open(debug_data_path, 'rb') as f:
            debug_data: list[DebugData] = pickle.load(f)

        analysis_mod.plot_right_wheel_act(settings, debug_data, right_wheel_act_file_path)

    # Plot the pheromone activity
    if not os.path.exists(pheromone_act_file_path):
        with open(debug_data_path, 'rb') as f:
            debug_data: list[DebugData] = pickle.load(f)

        analysis_mod.plot_pheromone_act(settings, debug_data, pheromone_act_file_path)

    # Plot the total gas pheromone
    if not os.path.exists(total_gas_pheromone_graph_path):
        with open(debug_data_path, 'rb') as f:
            debug_data: list[DebugData] = pickle.load(f)

        analysis_mod.plot_total_gas_pheromone(settings, debug_data, total_gas_pheromone_graph_path)

    # Plot the total liquid pheromone
    if not os.path.exists(total_liquid_pheromone_graph_path):
        with open(debug_data_path, 'rb') as f:
            debug_data: list[DebugData] = pickle.load(f)

        analysis_mod.plot_total_liquid_pheromone(settings, debug_data, total_liquid_pheromone_graph_path)


def main(settings: Settings):
    save_dir = os.path.abspath(
        analysis_mod.utils.get_latest_folder(settings.Storage.SAVE_DIRECTORY)
    )
    # save_dir = "./results/20250829-153444_bf9923e9"

    saved_individuals = IndividualRecorder.load(
        os.path.join(save_dir, "optimization_log.pkl")
    )
    fittness_graph_path = os.path.join(save_dir, "loss_history.png")
    pheromone_graph_path = os.path.join(save_dir, "pheromone_history.png")

    analysis_mod.collect_loss(save_dir, settings, saved_individuals, SimulatorForDebugging)
    analysis_mod.plot_fitness(save_dir, fittness_graph_path)
    analysis_mod.plot_pheromone_history(pheromone_graph_path, saved_individuals)

    # Analyze a specific individual from a specific generation (e.g., generation 0)
    for g in [0, -1]:
        rec: Rec = saved_individuals[g]
        individual: Individual = rec.best_individual
        analyze_specific_individual(save_dir, settings, individual, seed=None)


if __name__ == '__main__':
    from settings import MySettings

    main(
        MySettings()
    )
