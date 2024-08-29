import mujoco

from libs.optimizer import CMAES, Hist, MultiThreadProc
from libs.utils.data_collector import Recorder

from .settings import Settings
from .neural_network import NeuralNetwork
from .task_generator import TaskGenerator


def optimization() -> Hist:
    dim = NeuralNetwork().num_dim()

    cmaes = CMAES(
        dim=dim,
        generation=Settings.Optimization.GENERATION,
        population=Settings.Optimization.POPULATION,
        sigma=Settings.Optimization.SIGMA,
        mu=Settings.Optimization.MU
    )
    for gen in range(1, 1 + cmaes.get_generation()):
        task_generator = TaskGenerator()
        cmaes.optimize_current_generation(task_generator, MultiThreadProc)

    return cmaes.get_history()


def analysis():
    pass


def record(para, workdir):
    generator = TaskGenerator()
    task = generator.generate(para)

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = 29

    total_step = int(Settings.Task.EPISODE / Settings.Simulation.TIMESTEP + 0.5)
    rec = Recorder(
        Settings.Simulation.TIMESTEP,
        total_step,
        Settings.Renderer.RESOLUTION[0],
        Settings.Renderer.RESOLUTION[1],
        workdir,
        camera,
        Settings.Renderer.MAX_GEOM
    )

    rec.run(task)
