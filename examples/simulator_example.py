import jax

from framework.prelude import *
from framework.utils import GenericTkinterViewer

from config import RandomPatternController, FoodRelocationSimulator


def jaxable_example():
    cpu_device = jax.devices("cpu")[0]

    settings = Settings()

    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    settings.Robot.NUM = 1
    settings.Robot.INITIAL_POSITION = []

    settings.Food.NUM = 1
    settings.Food.INITIAL_POSITION = []

    settings.Pheromone.CELL_SIZE = 0.5
    settings.Pheromone.WIDTH_NUM = int(settings.Simulation.WORLD_WIDTH / settings.Pheromone.CELL_SIZE)
    settings.Pheromone.HEIGHT_NUM = int(settings.Simulation.WORLD_HEIGHT / settings.Pheromone.CELL_SIZE)

    rngs = jax.random.PRNGKey(0)
    mj_model, backend = FoodRelocationSimulator.new(
        settings,
        RandomPatternController(settings.Robot.NUM),
        rngs
    )
    backend = jax.device_put(backend, cpu_device)

    viewer = GenericTkinterViewer(mj_model, settings, backend)
    viewer.run()


if __name__ == '__main__':
    jaxable_example()
