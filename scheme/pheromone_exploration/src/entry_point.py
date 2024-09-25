import os
import json

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from libs.utils.data_collector import Recorder

from .settings import Settings
from .rec_env import RecEnv2
from .collector import IncreaseData2, DecreaseData2


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


def dump(case_dir, para):
    data_inc = IncreaseData2(para)
    data_dec = DecreaseData2(data_inc)

    np.save(os.path.join(case_dir, "inc_gas.npy"), data_inc.gas)
    np.save(os.path.join(case_dir, "evaporation.npy"), data_inc.evaporation)
    np.save(os.path.join(case_dir, "inc_gas.npy"), data_dec.gas)
    with open(os.path.join(case_dir, "parameter.json"), mode="w", encoding="utf-8") as f:
        json.dump(para, f, ensure_ascii=False, indent=2)

    return data_inc, data_dec


def analysis2(case_dir, data_inc: IncreaseData2, data_dec: DecreaseData2):
    sv = data_inc.sv

    def plot(name: str, yx, start=0, end=None, title=None):
        start_index = int(start / Settings.Simulation.TIMESTEP)
        end_index = Settings.Simulation.TOTAL_STEP if end is None else int(end / Settings.Simulation.TIMESTEP + 0.5)
        x = (np.arange(0, end_index - start_index) + 0.5) * Settings.Simulation.TIMESTEP + Settings.Plot.START
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.plot(x, yx[start_index:end_index])
        if title is not None:
            axis.set_title(title)
        fig.savefig(os.path.join(case_dir, name))

    plot("evaporation.svg", data_inc.evaporation)
    plot("gas_vol_inc.svg", np.max(data_inc.gas, axis=(1, 2)))
    plot("gas_vol_dec.svg", np.max(data_dec.gas, axis=(1, 2)))

    center_idx = Settings.Environment.CENTER_INDEX
    size = int(Settings.Plot.AT_POINT / Settings.Environment.CELL_SIZE)
    g1 = data_inc.gas[:, center_idx[0], center_idx[1] + int(size)]
    g2 = data_inc.gas[:, center_idx[0], center_idx[1] + int(size) + 1]
    gas = ((g2 - g1) * (size - int(size)) + g1) / sv
    plot("gas_vol_at_the_point.svg", gas)
