from ._mujoco_utils import (
    axisangle_to_quat,
    add_body, add_geom, add_joint, add_site, add_velocity_actuator, add_velocimeter, add_position_actuator
)
from ..prelude import *


def add_wall(spec: mujoco.MjSpec, settings: Settings) -> None:
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


def add_nest(spec: mujoco.MjSpec, settings: Settings) -> mujoco._specs.MjsSite:
    return add_site(
        spec.worldbody,
        name="nest",
        pos=(
            settings.Nest.POSITION.x,
            settings.Nest.POSITION.y,
            -settings.Nest.HEIGHT * 0.5 - 0.001
        ),
        size=[settings.Nest.RADIUS, settings.Nest.HEIGHT * 0.5, 0],
        rgba=settings.Nest.COLOR,  # green
        type_=mujoco.mjtGeom.mjGEOM_CYLINDER
    )


def add_food_object(spec: mujoco.MjSpec, settings: Settings, id_: int, position: Position) -> FoodSpec:
    """Add a food object to the simulation environment.
    
    Creates a cylindrical food object with physics properties, joints for
    movement, and velocity sensors for tracking.
    
    Args:
        spec: MuJoCo simulation specification
        settings: Settings object containing food properties
        id_: Unique identifier for the food object
        position: Position to place the food object
        
    Raises:
        ValueError: If any required parameter is None or invalid
        TypeError: If id_ is not an integer
    """
    if spec is None:
        raise ValueError("MuJoCo specification cannot be None")
    if settings is None:
        raise ValueError("Settings cannot be None")
    if position is None:
        raise ValueError("Position cannot be None")
    if not isinstance(id_, int):
        raise TypeError("Food ID must be an integer")
    if id_ < 0:
        raise ValueError("Food ID must be non-negative")
    food_body = add_body(
        spec.worldbody,
        pos=(position.x, position.y, settings.Food.HEIGHT * 0.5),
    )
    add_geom(
        food_body,
        geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(settings.Food.RADIUS, settings.Food.HEIGHT * 0.5, 0),
        rgba=settings.Food.COLOR,
        condim=FOOD_COLLISION_CONDIM,
        density=settings.Food.DENSITY
    )

    free_joint = add_joint(
        food_body,
        name=f"food{id_}_free",
        joint_type=mujoco.mjtJoint.mjJNT_FREE,
    )

    center_site = add_site(
        food_body,
        name=f"food{id_}_center",
    )

    velocimeter = add_velocimeter(
        spec,
        name=f"food{id_}_vel",
        site=center_site
    )

    return FoodSpec(
        center_site=center_site,
        free_join=free_joint,
        velocimeter=velocimeter
    )


def add_robot(
        spec: mujoco.MjSpec,
        settings: Settings,
        id_: int,
        robot_location: RobotLocation,
) -> RobotSpec:
    """Add a robot to the simulation environment.
    
    Creates a cylindrical robot with sliding and rotational joints,
    velocity actuators for movement control, and visual markers.
    
    Args:
        spec: MuJoCo simulation specification
        settings: Settings object containing robot properties
        id_: Unique identifier for the robot
        robot_location: Initial location and orientation of the robot
        
    Returns:
        Created robot body specification object
        
    Raises:
        ValueError: If any required parameter is None or invalid
        TypeError: If id_ is not an integer
        
    Note:
        Actuator velocity gain (kv) is taken from settings.Robot.ACTUATOR_KV
        if available, otherwise defaults to DEFAULT_ACTUATOR_KV.
    """
    if spec is None:
        raise ValueError("MuJoCo specification cannot be None")
    if settings is None:
        raise ValueError("Settings cannot be None")
    if robot_location is None:
        raise ValueError("Robot location cannot be None")
    if not isinstance(id_, int):
        raise TypeError("Robot ID must be an integer")
    if id_ < 0:
        raise ValueError("Robot ID must be non-negative")

    robot_body = add_body(
        spec.worldbody,
        name=f"robot{id_}",
        pos=(robot_location.x, robot_location.y, settings.Robot.HEIGHT * 0.5),
        quat=axisangle_to_quat((0, 0, 1), robot_location.theta),
    )

    add_geom(
        robot_body,
        geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(settings.Robot.RADIUS, settings.Robot.HEIGHT * 0.5, 0),
        rgba=settings.Robot.COLOR,
        condim=ROBOT_COLLISION_CONDIM,
        mass=settings.Robot.MASS,
    )

    center_site = add_site(
        robot_body,
        name=f"robot{id_}_center",
        pos=(0, 0, 0),
    )

    front_site_size = settings.Robot.RADIUS * ROBOT_FRONT_SIZE_FACTOR
    front_site = add_site(
        robot_body,
        name=f"robot{id_}_front",
        pos=(0, settings.Robot.RADIUS * ROBOT_FRONT_POSITION_FACTOR, front_site_size),
        size=[front_site_size, front_site_size, front_site_size],
        rgba=ROBOT_FRONT_COLOR,
        type_=mujoco.mjtGeom.mjGEOM_SPHERE
    )

    free_join = add_joint(
        robot_body,
        name=f"robot{id_}_joint",
        joint_type=mujoco.mjtJoint.mjJNT_FREE,
    )

    x_act = add_velocity_actuator(
        spec,
        name=f"robot{id_}_x_act",
        joint=free_join,
        kv=settings.Robot.ACTUATOR_MOVE_KV,
        gear=(1, 0, 0, 0, 0, 0)
    )

    y_act = add_velocity_actuator(
        spec,
        name=f"robot{id_}_y_act",
        joint=free_join,
        kv=settings.Robot.ACTUATOR_MOVE_KV,
        gear=(0, 1, 0, 0, 0, 0)
    )

    z_act = add_position_actuator(
        spec,
        joint=free_join,
        kp=50,
        kv=1,
        name=f"robot{id_}_z_act",
        gear=(0, 0, 1, 0, 0, 0)
    )

    r_act = add_velocity_actuator(
        spec,
        name=f"robot{id_}_r_act",
        joint=free_join,
        kv=settings.Robot.ACTUATOR_ROT_KV,
        gear=(0, 0, 0, 0, 0, 1)
    )

    return RobotSpec(
        center_site=center_site,
        front_site=front_site,
        free_join=free_join,
        x_act=x_act,
        y_act=y_act,
        z_act=z_act,
        r_act=r_act
    )
