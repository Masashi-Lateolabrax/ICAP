"""Environment setup and creation for MuJoCo simulation."""

import mujoco

from .mujoco_core import add_texture, add_material, add_geom, add_site
from .mujoco_objects import add_food_object, add_robot
from .mujoco_constants import (
    CHECKER_TEXTURE_SIZE, CHECKER_RGB_WHITE, CHECKER_RGB_GRAY, WALL_COLLISION_CONDIM
)
from ..config.settings import Settings, RobotLocation, Position
from .position_generators import rand_food_pos, rand_robot_pos


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
        texture=["simple_checker"],
        texrepeat=(
            settings.Simulation.WORLD_WIDTH * 0.5,
            settings.Simulation.WORLD_HEIGHT * 0.5
        )
    )


def setup_wall(spec: mujoco.MjSpec, settings: Settings) -> None:
    """Setup boundary walls around the simulation area.
    
    Creates four box-shaped walls (North, South, East, West) that form
    a rectangular boundary around the simulation area.
    
    Args:
        spec: MuJoCo simulation specification
        settings: Settings object containing area and wall dimensions
    """
    # Calculate total dimensions including wall thickness
    w = settings.Simulation.WORLD_WIDTH + settings.Simulation.WALL_THICKNESS
    h = settings.Simulation.WORLD_HEIGHT + settings.Simulation.WALL_THICKNESS

    # Wall thickness for convenience
    t = settings.Simulation.WALL_THICKNESS * 0.5

    for name, pos_x, pos_y, size_x, size_y in [
        ("wallN", 0, h * 0.5, w * 0.5, t),
        ("wallS", 0, h * -0.5, w * 0.5, t),
        ("wallW", w * 0.5, 0, t, h * 0.5),
        ("wallE", w * -0.5, 0, t, h * 0.5),
    ]:
        add_geom(
            spec.worldbody,
            name=name,
            geom_type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(pos_x, pos_y, settings.Simulation.WALL_HEIGHT * 0.5),
            size=(size_x, size_y, settings.Simulation.WALL_HEIGHT * 0.5),
            condim=WALL_COLLISION_CONDIM
        )


def setup_markers(spec: mujoco.MjSpec, settings: Settings) -> None:
    add_site(
        spec.worldbody,
        name="nest",
        pos=(
            settings.Nest.POSITION.x,
            settings.Nest.POSITION.y,
            -settings.Nest.HEIGHT * 0.5
        ),
        size=[settings.Nest.RADIUS, settings.Nest.HEIGHT * 0.5, 0],
        rgba=settings.Nest.COLOR,  # green
        type_=mujoco.mjtGeom.mjGEOM_CYLINDER
    )


def setup_objects(spec: mujoco.MjSpec, settings: Settings) -> None:
    setup_wall(spec, settings)

    add_geom(
        spec.worldbody,
        geom_type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0, 0, 0),
        size=(
            settings.Simulation.WORLD_WIDTH * 0.5,
            settings.Simulation.WORLD_HEIGHT * 0.5,
            1
        ),
        material="ground",
    )

    invalid_area: list[tuple[Position, float]] = []

    # Create food objects
    for i in range(settings.Food.NUM):
        if i < len(settings.Food.INITIAL_POSITION):
            position: Position = settings.Food.INITIAL_POSITION[i]
        else:
            position: Position = rand_food_pos(settings, invalid_area)
            invalid_area.append(
                (position, settings.Food.RADIUS)
            )
        add_food_object(spec, settings, i, position)

    # Create robots
    for i in range(settings.Robot.NUM):
        if i < len(settings.Robot.INITIAL_POSITION):
            position: RobotLocation = settings.Robot.INITIAL_POSITION[i]
        else:
            position: RobotLocation = rand_robot_pos(settings, invalid_area)
            invalid_area.append(
                (position.position, settings.Robot.RADIUS)
            )

        add_robot(spec, settings, i, position)


def create_environment(
        settings: Settings,
) -> mujoco.MjSpec:
    """Create a complete MuJoCo simulation environment.
    
    Assembles a simulation environment with ground plane, nest, walls,
    and configurable numbers of robots and food objects.
    
    Args:
        settings: Settings object containing all environment parameters
        num_robots: Number of robots to create (default: 1)
        num_food: Number of food objects to create (default: 1)
        robot_positions: Optional list of robot positions. If None, places robots at origin.
        food_positions: Optional list of food positions. If None, places food at origin.
        
    Returns:
        Complete MuJoCo simulation specification ready for compilation
        
    Raises:
        ValueError: If settings is None, invalid, or position list lengths don't match counts
        
    Note:
        If position lists are provided, their length must match the corresponding count.
        If not provided, all objects are placed at origin (0, 0) with default orientation.
    """
    if settings is None:
        raise ValueError("Settings cannot be None")

    if not hasattr(settings, 'Simulation') or not hasattr(settings, 'Robot') or not hasattr(settings, 'Food'):
        raise ValueError("Settings must contain Simulation, Robot, and Food configurations")

    spec = mujoco.MjSpec()

    setup_option(spec, settings)
    # setup_visual(spec, settings)
    setup_textures(spec, settings)
    setup_markers(spec, settings)
    setup_objects(spec, settings)

    return spec
