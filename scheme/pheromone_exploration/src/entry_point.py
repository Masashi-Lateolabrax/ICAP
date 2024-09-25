import os

import mujoco
import numpy as np

from libs.optimizer import CMAES, Hist
from libs.utils.data_collector import Recorder

from .settings import Settings
from .task import Generator
from .rec_env import RecEnv, RecEnv2
from .utils import convert_para
from .collector import IncreaseData, DecreaseData, IncreaseData2, DecreaseData2


def optimization() -> Hist:
    generator = Generator()
    cmaes = CMAES(
        dim=4,
        generation=Settings.Optimization.GENERATION,
        population=Settings.Optimization.POPULATION,
        mu=Settings.Optimization.NUM_ELITE,
        sigma=Settings.Optimization.SIGMA,
        split_tasks=1
    )
    cmaes.optimize(generator)
    hist = cmaes.get_history()

    best = hist.get_min()
    para = convert_para(best.min_para)
    for k, v in para.items():
        print(f"{k}: {v}")

    return hist


def recode(para, gas_inc: IncreaseData, gas_dec: DecreaseData, workdir):
    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = Settings.Display.ZOOM
    recorder = Recorder(
        timestep=Settings.Simulation.TIMESTEP,
        episode=int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5),
        width=Settings.Display.RESOLUTION[0],
        height=Settings.Display.RESOLUTION[1],
        project_directory=workdir,
        camera=camera,
        max_geom=Settings.Display.MAX_GEOM
    )

    recorder.run(
        RecEnv(gas_inc.gas, para)
    )
    recorder.run(
        RecEnv(gas_dec.gas, para)
    )
    recorder.release()


def record(gas_inc: IncreaseData2, gas_dec: DecreaseData2, case_dir):
    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = Settings.Display.ZOOM
    recorder = Recorder(
        timestep=Settings.Simulation.TIMESTEP,
        episode=int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5),
        width=Settings.Display.RESOLUTION[0],
        height=Settings.Display.RESOLUTION[1],
        project_directory=case_dir,
        camera=camera,
        max_geom=Settings.Display.MAX_GEOM
    )

    recorder.run(
        RecEnv2(gas_inc.gas, gas_inc.sv)
    )
    recorder.run(
        RecEnv2(gas_dec.gas, gas_dec.sv)
    )
    recorder.release()


def analysis(workdir, para, data_inc: IncreaseData, data_dec: DecreaseData):
    import matplotlib.pyplot as plt

    start_index = int(Settings.Plot.START / Settings.Simulation.TIMESTEP)
    end_index = int(Settings.Plot.END / Settings.Simulation.TIMESTEP + 0.5)
    sv = convert_para(para)["sv"]

    # Evaporation Speed
    x = (np.arange(0, end_index - start_index) + 0.5) * Settings.Simulation.TIMESTEP + Settings.Plot.START

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(x, data_inc.dif_liquid[start_index:end_index])
    fig.savefig(os.path.join(workdir, "evaporation_speed.svg"))

    # Evaporation Speed Extended
    x = (np.arange(0, data_inc.dif_liquid.shape[0]) + 0.5) * Settings.Simulation.TIMESTEP
    base_speed = data_inc.dif_liquid[end_index]
    ave_speed = np.average(data_inc.dif_liquid[end_index:])

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(f"base: {base_speed}, ave: {ave_speed}")
    axis.plot(x, data_inc.dif_liquid)
    fig.savefig(os.path.join(workdir, "evaporation_speed_ex.svg"))

    # Gas Volume (INCREASE)
    x = (np.arange(0, end_index - start_index) + 0.5) * Settings.Simulation.TIMESTEP + Settings.Plot.START
    gas_max = np.max(data_dec.gas[start_index: end_index]) / sv

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(f"max: {gas_max}")
    axis.plot(x, np.max(data_inc.gas[start_index:end_index], axis=(1, 2)) / sv)
    fig.savefig(os.path.join(workdir, "gas_volume_inc.svg"))

    # Gas Volume Extended (INCREASE)
    x = (np.arange(0, data_inc.gas.shape[0]) + 0.5) * Settings.Simulation.TIMESTEP
    gas_max = np.max(data_dec.gas) / sv

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(f"max: {gas_max}")
    axis.plot(x, np.max(data_inc.gas, axis=(1, 2)) / sv)
    fig.savefig(os.path.join(workdir, "gas_volume_inc_ex.svg"))

    # Gas Volume (DECREASE)
    x = (np.arange(0, end_index - start_index) + 0.5) * Settings.Simulation.TIMESTEP + Settings.Plot.START
    gas_max = np.max(data_dec.gas[start_index:end_index])
    gas_min = np.min(data_dec.gas[start_index:end_index]) / sv
    half_life_idx = np.min(np.where(np.max(data_dec.gas[start_index:end_index], axis=(1, 2)) < gas_max * 0.5)[0])
    half_life_time = half_life_idx * Settings.Simulation.TIMESTEP

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(f"min: {gas_min}, half life time: {half_life_time}")
    axis.plot(x, np.max(data_dec.gas[start_index:end_index], axis=(1, 2)) / sv)
    fig.savefig(os.path.join(workdir, "gas_volume_dec.svg"))

    # Gas Volume Extended (DECREASE)
    x = (np.arange(0, data_dec.gas.shape[0]) + 0.5) * Settings.Simulation.TIMESTEP
    gas_max = np.max(data_dec.gas)
    gas_min = np.min(data_dec.gas) / sv
    half_life_idx = np.min(np.where(np.max(data_dec.gas, axis=(1, 2)) < gas_max * 0.5)[0])
    half_life_time = half_life_idx * Settings.Simulation.TIMESTEP

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(f"min: {gas_min}, half life time: {half_life_time}")
    axis.plot(x, np.max(data_dec.gas, axis=(1, 2)) / sv)
    fig.savefig(os.path.join(workdir, "gas_volume_dec_ex.svg"))

    # Gas Volume at The Point
    x = (np.arange(0, end_index - start_index) + 0.5) * Settings.Simulation.TIMESTEP + Settings.Plot.START
    size = Settings.Optimization.Target.FIELD_SIZE / Settings.Pheromone.CELL_SIZE_FOR_MUJOCO
    center_idx = Settings.Pheromone.CENTER_INDEX
    g1 = data_inc.gas[start_index:end_index, center_idx[0], center_idx[1] + int(size)]
    g2 = data_inc.gas[start_index:end_index, center_idx[0], center_idx[1] + int(size) + 1]
    gas = ((g2 - g1) * (size - int(size)) + g1) / sv
    gas_max = np.max(gas)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(f"Gas Volume at {Settings.Optimization.Target.FIELD_SIZE}, max: {gas_max}")
    axis.plot(x, gas)
    fig.savefig(os.path.join(workdir, "gas_volume_at_the_point.svg"))

    # Relative Gas Volume at The Point
    x = (np.arange(0, end_index - start_index) + 0.5) * Settings.Simulation.TIMESTEP + Settings.Plot.START
    size = Settings.Optimization.Target.FIELD_SIZE / Settings.Pheromone.CELL_SIZE_FOR_MUJOCO
    center_idx = Settings.Pheromone.CENTER_INDEX
    gc = data_inc.gas[start_index:end_index, center_idx[0], center_idx[1]]
    gc[gc == 0] = 1
    g1 = data_inc.gas[start_index:end_index, center_idx[0], center_idx[1] + int(size)]
    g2 = data_inc.gas[start_index:end_index, center_idx[0], center_idx[1] + int(size) + 1]
    gas = ((g2 - g1) * (size - int(size)) + g1) / gc
    gas_max = np.max(gas)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(f"Gas Volume at {Settings.Optimization.Target.FIELD_SIZE}, max: {gas_max}")
    axis.plot(x, gas)
    fig.savefig(os.path.join(workdir, "relative_gas_volume_at_the_point.svg"))

    # Relative Gas Volume at The Point Extended
    x = (np.arange(0, data_inc.dif_liquid.shape[0]) + 0.5) * Settings.Simulation.TIMESTEP
    size = Settings.Optimization.Target.FIELD_SIZE / Settings.Pheromone.CELL_SIZE_FOR_MUJOCO
    center_idx = Settings.Pheromone.CENTER_INDEX
    gc = data_inc.gas[:, center_idx[0], center_idx[1]]
    gc[gc == 0] = 1
    g1 = data_inc.gas[:, center_idx[0], center_idx[1] + int(size)]
    g2 = data_inc.gas[:, center_idx[0], center_idx[1] + int(size) + 1]
    gas = ((g2 - g1) * (size - int(size)) + g1) / gc
    gas_max = np.max(gas)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(f"Gas Volume at {Settings.Optimization.Target.FIELD_SIZE}, max: {gas_max}")
    axis.plot(x, gas)
    fig.savefig(os.path.join(workdir, "relative_gas_volume_at_the_point_ex.svg"))
