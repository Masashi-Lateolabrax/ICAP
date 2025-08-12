from abc import ABC
import os

import mujoco

from ..prelude import *
from .basic import BasicMuJoCoSimulator

from ..environment import (
    add_geom,
    setup_option, setup_visual, setup_textures, add_nest, add_wall,
    rand_robot_pos, rand_food_pos, add_mesh_in_asset, MeshContentType,
    add_food_object_with_mesh, add_robot_with_mesh
)


def generate_mjspec(
        settings: Settings
) -> tuple[mujoco.MjSpec, mujoco._specs.MjsSite, list[RobotSpec], list[FoodSpec]]:
    spec = mujoco.MjSpec()

    setup_option(spec, settings)
    setup_visual(spec, settings)
    setup_textures(spec, settings)

    add_wall(spec, settings)

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
        rgba=GROUND_COLOR,
        condim=GROUND_COLLISION_CONDIM
    )

    nest_spec = add_nest(spec, settings)

    invalid_area: list[tuple[Position, float]] = []

    # Create food objects
    food_specs = []
    if settings.Food.NUM > 0:
        food_mesh = add_mesh_in_asset(
            spec,
            name="food_mesh",
            file=os.path.abspath(os.path.join(settings.Storage.ASSET_DIRECTORY, "food-object.stl")),
            content_type=MeshContentType.STL,
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        )

        for i in range(settings.Food.NUM):
            if i < len(settings.Food.INITIAL_POSITION):
                position: Position = settings.Food.INITIAL_POSITION[i]
            else:
                position: Position = rand_food_pos(settings, invalid_area)
            invalid_area.append(
                (position, settings.Food.RADIUS)
            )
            food_specs.append(
                add_food_object_with_mesh(spec, settings, food_mesh, i, position)
            )

    # Create robots
    robot_specs = []
    if settings.Robot.NUM > 0:
        robot_mesh = add_mesh_in_asset(
            spec,
            name="robot_mesh",
            file=os.path.abspath(os.path.join(settings.Storage.ASSET_DIRECTORY, "robot-object.stl")),
            content_type=MeshContentType.STL,
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        )

        for i in range(settings.Robot.NUM):
            if i < len(settings.Robot.INITIAL_POSITION):
                position: RobotLocation = settings.Robot.INITIAL_POSITION[i]
            else:
                position: RobotLocation = rand_robot_pos(settings, invalid_area)
            invalid_area.append(
                (position.position, settings.Robot.RADIUS)
            )
            robot_specs.append(
                add_robot_with_mesh(spec, settings, robot_mesh, i, position)
            )

    return spec, nest_spec, robot_specs, food_specs


class BasicSimulator(BasicMuJoCoSimulator, ABC):
    def __init__(self, settings, render: bool = False):
        mj_spec, nest_spec, robot_specs, food_specs = generate_mjspec(settings)
        super().__init__(settings, mj_spec, render)

        self.nest_site = self.data.site(nest_spec.name)
        self.robot_values = [
            RobotValues(settings.Robot.DISTANCE_BETWEEN_WHEELS, settings.Robot.MAX_SPEED, self.data, s)
            for s in robot_specs
        ]
        self.food_values = [FoodValues(self.data, s) for s in food_specs]
