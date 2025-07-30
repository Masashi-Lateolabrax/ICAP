import mujoco

from ._mujoco_utils import (
    add_texture, add_material,
)
from ..prelude import *


def setup_option(spec: mujoco.MjSpec, settings: Settings) -> None:
    """Configure simulation options from settings.

    Sets up timestep, integrator, and contact cone configuration.

    Args:
        spec: MuJoCo simulation specification
        settings: Settings object containing simulation parameters
    """
    spec.option.timestep = settings.Simulation.TIME_STEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_RK4  # Runge-Kutta 4th order integrator
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC


def setup_textures(spec: mujoco.MjSpec, settings: Settings) -> None:
    """Setup default textures and materials for the simulation.

    Creates a checker texture pattern and ground material based on
    simulation area dimensions from settings.

    Args:
        spec: MuJoCo simulation specification
        settings: Settings object containing area dimensions

    Raises:
        ValueError: If spec or settings is None
    """
    if spec is None:
        raise ValueError("MuJoCo specification cannot be None")
    if settings is None:
        raise ValueError("Settings cannot be None")
    add_texture(
        spec,
        name="simple_checker",
        type_=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=CHECKER_TEXTURE_SIZE,
        height=CHECKER_TEXTURE_SIZE,
        rgb1=CHECKER_RGB_WHITE,
        rgb2=CHECKER_RGB_GRAY
    )
    add_material(
        spec,
        name="ground",
        texture="simple_checker",
        texrepeat=(
            settings.Simulation.WORLD_WIDTH * 0.5,
            settings.Simulation.WORLD_HEIGHT * 0.5
        )
    )


def setup_visual(spec: mujoco.MjSpec, settings: Settings) -> None:
    visual: mujoco._specs.MjVisual = spec.visual
    visual.global_.offwidth = settings.Render.RENDER_WIDTH
    visual.global_.offheight = settings.Render.RENDER_HEIGHT
